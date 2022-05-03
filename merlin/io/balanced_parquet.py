import functools

import dask.dataframe as dd
import fsspec.parquet as fsspec_parquet
import pyarrow as pa
import pyarrow.dataset as pa_ds
import pyarrow.parquet as pq
from dask.utils import natural_sort_key

from merlin.core.utils import run_on_worker
from merlin.io.dataset_engine import DatasetEngine
from merlin.io.parquet import _sample_row_group
from merlin.io.shuffle import shuffle_df


class _ReadTask:
    def __init__(self, task_size):
        self.task_size = task_size
        self.paths = []
        self.groups = []
        self.starts = []
        self.stops = []
        self.sizes = []

    @property
    def size(self):
        return sum(self.sizes)

    def to_tuple(self):
        if not self.paths:
            return None

        global_start = self.starts[0]
        global_stop = global_start + self.size
        return (
            self.paths,
            self.groups,
            global_start,
            global_stop,
        )

    def coalesce(self, path, group, start, stop):

        group = group if isinstance(group, list) else [group]
        if self.paths and self.paths[-1] == path and self.groups[-1][-1] < group[0]:
            self.groups[-1] += group
            self.stops[-1] = stop
            self.sizes[-1] += stop - start
        else:
            self.paths.append(path)
            self.groups.append(group)
            self.starts.append(start)
            self.stops.append(stop)
            self.sizes.append(stop - start)

    def add_read(self, path, group, start, stop):

        rows_allowed = self.task_size - self.size

        if not rows_allowed:
            return (path, group, start, stop)

        num_rows = stop - start
        if num_rows > rows_allowed:
            # We CANNOT fit this entire row-group
            new_stop = start + rows_allowed
            self.coalesce(path, group, start, new_stop)
            return (path, group, new_stop, stop)
        else:
            # We CAN read the entire row-group
            self.coalesce(path, group, start, stop)
            return None


class DatasetInfo:
    def __init__(
        self,
        path,
        sort=True,
        filesystem=None,
        filters=None,
        partitioning="hive",
        **dataset_options,
    ):
        self.dataset = pa_ds.dataset(
            path,
            filesystem=filesystem,
            format="parquet",
            partitioning=partitioning,
            **dataset_options,
        )
        self.filters = filters
        self.stats = self._get_dataset_stats()
        if sort:
            self._sort_by_path()

    def _get_dataset_stats(self):
        return pa.concat_tables(
            [self._get_fragment_stats(frag) for frag in self.dataset.get_fragments(self.filters)]
        ).to_pandas()

    def _get_fragment_stats(self, frag):
        row_counts = [row_group.num_rows for row_group in frag.row_groups]
        rg_ids = [row_group.id for row_group in frag.row_groups]
        size = len(row_counts)
        return pa.table(
            [
                pa.array([frag.path] * size),
                pa.array([str(frag.partition_expression)] * size),
                pa.array(rg_ids),
                pa.array(row_counts),
            ],
            names=["path", "partition", "group", "num_rows"],
        )

    def _sort_by_path(self):
        self.stats["__sort_key__"] = self.stats.path + "." + self.stats.group.astype(str)
        self.stats = self.stats.sort_values(
            "__sort_key__",
            key=lambda x: x.apply(natural_sort_key),
            ignore_index=True,
        ).drop(columns=["__sort_key__"])

    def generate_tasks(self, task_size, shuffle=False, drop_residual=True):
        read_tasks = []
        stats = shuffle_df(self.stats) if shuffle else self.stats
        for i in range(len(stats)):

            if not read_tasks:
                read_tasks.append(_ReadTask(task_size))

            record = stats.iloc[i]
            residual_read = read_tasks[-1].add_read(
                record["path"], record["group"], 0, record["num_rows"]
            )
            while residual_read:
                read_tasks.append(_ReadTask(task_size))
                residual_read = read_tasks[-1].add_read(*residual_read)

        if drop_residual and read_tasks[-1].size < task_size:
            read_tasks.pop(-1)

        return [read_task.to_tuple() for read_task in read_tasks]

    def sample_task(self):
        sample = self.stats.iloc[0]
        return (sample["path"], sample["group"], None, None)


class _PartitionReader:
    def __init__(self, fs, cpu=True, **dataset_options):
        self.fs = fs
        self.cpu = cpu
        self.dataset_options = dataset_options
        self._columns = None

    @property
    def columns(self):
        return self._columns

    def project_columns(self, columns):
        self._columns = columns
        return self

    def __call__(self, task, shuffle=False, **read_kwargs):
        paths, groups, global_start, global_stop = task

        if not self.cpu:
            import cudf

            if cudf.utils.ioutils._is_local_filesystem(self.fs):
                df = cudf.read_parquet(paths, index=False, row_groups=groups, **read_kwargs).iloc[
                    global_start:global_stop
                ]
            else:
                # TODO: Can we do this faster?
                dfs = []
                for path, row_groups in zip(paths, groups):
                    rgs = row_groups if isinstance(row_groups, list) else [row_groups]
                    with fsspec_parquet.open_parquet_file(
                        path, columns=self.columns, row_groups=rgs
                    ) as fil:
                        dfs.append(
                            cudf.read_parquet(
                                fil, columns=self.columns, row_groups=rgs, **read_kwargs
                            )
                        )
                df = cudf.concat(dfs).iloc[global_start:global_stop]
        else:
            tables = []
            if not isinstance(paths, list):
                paths = [paths]
            if not isinstance(groups, list):
                groups = [groups]
            for path, row_groups in zip(paths, groups):
                rgs = row_groups if isinstance(row_groups, list) else [row_groups]
                with fsspec_parquet.open_parquet_file(
                    path, columns=self.columns, row_groups=rgs
                ) as fil:
                    tables.append(
                        pq.ParquetFile(fil).read_row_groups(
                            rgs,
                            columns=self.columns,
                            use_threads=False,
                            use_pandas_metadata=False,
                            **read_kwargs,
                        )
                    )
            df = pa.concat_tables(tables).to_pandas().iloc[global_start:global_stop]

        if shuffle:
            return shuffle_df(df)
        return df


class BalancedParquetEngine(DatasetEngine):
    """Parquet-based DatasetEngine with deterministic partition sizes"""

    def __init__(
        self,
        paths,
        part_size,
        storage_options,
        rows_per_partition=None,
        batch_size=None,
        cpu=False,
        dataset_options=None,
        drop_residual=True,
        **kwargs,
    ):
        super().__init__(paths, part_size, cpu=cpu, storage_options=storage_options)
        self._pp_nrows = None
        self._real_meta = None
        self._cached_tasks = None

        self.rows_per_partition = rows_per_partition
        self.drop_residual = drop_residual
        self.read_parquet_kwargs = kwargs.copy()
        self.dataset_options = dataset_options or {}

        if self.rows_per_partition is None:
            self._real_meta, rg_byte_size_0 = run_on_worker(
                _sample_row_group,
                self._path0,
                self.fs,
                cpu=self.cpu,
                memory_usage=True,
                **self.read_parquet_kwargs,
            )
            row_groups_per_part = self.part_size / rg_byte_size_0
            rows_per_part = self._size0 * int(row_groups_per_part)

            # Align partition with batch size if one was specified
            if batch_size:
                nbatches = rows_per_part // batch_size
                rows_per_part = batch_size * nbatches

        self.rows_per_partition = int(rows_per_part)

        assert self.rows_per_partition > 0

    def generate_tasks(self, shuffle=False):
        tasks = self._dataset_info.generate_tasks(
            self.rows_per_partition,
            shuffle=shuffle,
            drop_residual=self.drop_residual,
        )
        self._pp_nrows = [task[-1] - task[-2] for task in tasks]
        return tasks

    @property
    def _path0(self):
        if not hasattr(self, "_sample_path"):
            sample = self._dataset_info.stats.iloc[0]
            self._sample_path = sample["path"]
            self._sample_size = sample["num_rows"]
        return self._sample_path

    @property
    def _size0(self):
        if not hasattr(self, "_sample_size "):
            sample = self._dataset_info.stats.iloc[0]
            self._sample_path = sample["path"]
            self._sample_size = sample["num_rows"]
        return self._sample_size

    @property  # type: ignore
    @functools.lru_cache(1)
    def _dataset_info(self):
        paths = self.stripped_paths
        fs = self.fs
        if len(paths) > 1:
            # This is a list of files
            return DatasetInfo(paths, filesystem=fs, **self.dataset_options)
        else:
            # This is a directory or a single file
            return DatasetInfo(paths[0], filesystem=fs, **self.dataset_options)

    @property
    def _partition_lens(self):
        return self._pp_nrows

    @property
    def num_rows(self):
        # TODO: Avoid parsing metadata once upstream dask
        # can get the length efficiently (in all practical cases)
        if self._partition_lens:
            return sum(self._partition_lens)
        return len(self.to_ddf().index)

    def to_ddf(self, columns=None, cpu=None, shuffle=False):

        # Check if we are using cpu or gpu backend
        cpu = self.cpu if cpu is None else cpu

        # Gather partition plan
        tasks = self.generate_tasks(shuffle)

        # Return a DataFrame collection
        func = _PartitionReader(self.fs, cpu=cpu)
        return dd.from_map(
            func,
            tasks,
            meta=self.sample_data(),
            enforce_metadata=False,
        )

    def to_cpu(self):
        self.cpu = True

    def to_gpu(self):
        self.cpu = False

    def sample_data(self, n=1):
        """Return a real data sample from the Dataset"""
        if self._real_meta is not None:
            # First check is we already cached a data sample
            # while calculating `row_groups_per_part`
            _len = len(self._real_meta)
            if _len >= n:
                if _len == n:
                    _real_meta = self._real_meta
                else:
                    _real_meta = self._real_meta.take(list(range(n)))
                # We can clear self._real_meta, because the data
                # will still be cached at the Dataset level
                self._real_meta = None
                return _real_meta

        # Real metadata sample is not cached - Sample from
        # the first row-group in the Dataset
        return run_on_worker(
            _sample_row_group,
            self._path0,
            self.fs,
            cpu=self.cpu,
            n=n,
            memory_usage=False,
            **self.read_parquet_kwargs,
        ).take(list(range(n)))
