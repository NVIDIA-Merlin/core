#
# Copyright (c) 2022, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import logging

import dask
import pandas as pd
from dask.core import flatten

import merlin.dtypes as md
from merlin.core.dispatch import concat_columns, is_list_dtype, list_val_dtype
from merlin.core.utils import (
    ensure_optimize_dataframe_graph,
    global_dask_client,
    set_client_deprecated,
)
from merlin.dag import ColumnSelector, Graph, Node
from merlin.dtypes.shape import DefaultShapes
from merlin.io.worker import clean_worker_cache

LOG = logging.getLogger("merlin")


class LocalExecutor:
    """
    An executor for running Merlin operator DAGs locally
    """

    def transform(
        self,
        transformable,
        graph,
        output_dtypes=None,
        additional_columns=None,
        capture_dtypes=False,
        strict=False,
    ):
        """
        Transforms a single dataframe (possibly a partition of a Dask Dataframe)
        by applying the operators from a collection of Nodes
        """
        nodes = []
        if isinstance(graph, Graph):
            nodes.append(graph.output_node)
        elif isinstance(graph, Node):
            nodes.append(graph)
        elif isinstance(graph, list):
            nodes = graph
        else:
            raise TypeError(
                f"LocalExecutor detected unsupported type of input for graph: {type(graph)}."
                " `graph` argument must be either a `Graph` object (preferred)"
                " or a list of `Node` objects (deprecated, but supported for backward "
                " compatibility.)"
            )

        # There's usually only one node, but it's possibly to pass multiple nodes for `fit`
        # If we have multiple, we concatenate their outputs into a single transformable
        output_data = None
        for node in nodes:
            transformed_data = self._execute_node(node, transformable, capture_dtypes, strict)
            output_data = self._combine_node_outputs(node, transformed_data, output_data)

        # If there are any additional columns that weren't produced by one of the supplied nodes
        # we grab them directly from the supplied input data. Normally this would happen on a
        # per-node basis, but offers a safety net for the multi-node case
        if additional_columns:
            output_data = concat_columns(
                [output_data, transformable[_get_unique(additional_columns)]]
            )

        return output_data

    def _execute_node(self, node, transformable, capture_dtypes=False, strict=False):
        upstream_outputs = self._run_upstream_transforms(
            node, transformable, capture_dtypes, strict
        )
        upstream_columns = self._append_addl_root_columns(node, transformable, upstream_outputs)
        transform_input = self._merge_upstream_columns(upstream_columns)
        transform_output = self._run_node_transform(node, transform_input, capture_dtypes, strict)
        return transform_output

    def _run_upstream_transforms(self, node, transformable, capture_dtypes=False, strict=False):
        upstream_outputs = []

        for upstream_node in node.parents_with_dependencies:
            node_output = self._execute_node(
                upstream_node,
                transformable,
                capture_dtypes=capture_dtypes,
                strict=strict,
            )
            if node_output is not None and len(node_output) > 0:
                upstream_outputs.append(node_output)

        return upstream_outputs

    def _append_addl_root_columns(self, node, transformable, upstream_outputs):
        node_input_cols = set(node.input_schema.column_names)
        addl_input_cols = set(node.dependency_columns.names)

        already_present = set()
        for upstream_tensors in upstream_outputs:
            for col in node_input_cols:
                if col in upstream_tensors:
                    already_present.add(col)

        root_columns = node_input_cols.union(addl_input_cols) - already_present

        if root_columns:
            upstream_outputs.append(transformable[list(root_columns)])

        return upstream_outputs

    def _merge_upstream_columns(self, upstream_outputs, merge_fn=concat_columns):
        combined_outputs = None
        seen_columns = set()
        for upstream_output in upstream_outputs:
            if combined_outputs is None or not len(combined_outputs):
                combined_outputs = upstream_output
                seen_columns = set(upstream_output.columns)
            else:
                new_columns = set(upstream_output.columns) - seen_columns
                if new_columns:
                    combined_outputs = merge_fn(
                        [combined_outputs, upstream_output[list(new_columns)]]
                    )
                    seen_columns.update(new_columns)
        return combined_outputs

    def _run_node_transform(self, node, input_data, capture_dtypes=False, strict=False):
        """
        Run the transform represented by the final node in the graph
        and check output dtypes against the output schema
        Parameters
        ----------
        node : Node
            Output node of the graph to execute
        input_data : Transformable
            Dataframe to run the graph ending with node on
        capture_dtypes : bool, optional
            Overrides the schema dtypes with the actual dtypes when True, by default False
        strict : bool, optional
            Raises error if the dtype of returned data doesn't match the schema, by default False
        Returns
        -------
        Transformable
            The output DataFrame or DictArray formed by executing the final node's transform
        Raises
        ------
        TypeError
            If the transformed output columns don't have the same dtypes
            as the output schema columns when `strict` is True
        RuntimeError
            If no DataFrame or DictArray is returned from the operator
        """
        if not node.op:
            return input_data

        try:
            # use input_columns to ensure correct grouping (subgroups)
            selection = node.input_columns.resolve(node.input_schema)
            transformed_data = node.op.transform(selection, input_data)

            if transformed_data is None:
                raise RuntimeError(f"Operator {node.op} didn't return a value during transform")
            elif capture_dtypes:
                self._capture_dtypes(node, transformed_data)
            elif strict and len(transformed_data):
                self._validate_dtypes(node, transformed_data)

            return transformed_data

        except Exception as exc:
            LOG.exception("Failed to transform operator %s", node.op)
            raise exc

    def _capture_dtypes(self, node, output_data):
        for col_name, output_col_schema in node.output_schema.column_schemas.items():
            output_data_schema = self._build_schema_from_data(
                output_data, col_name, output_col_schema
            )
            node.output_schema.column_schemas[col_name] = output_data_schema

    # TODO: Turn this into a function
    def _build_schema_from_data(self, output_data, col_name, output_col_schema):
        column = output_data[col_name]
        column_dtype = column.dtype
        col_shape = output_col_schema.shape
        is_list = is_list_dtype(column)

        if is_list:
            column_dtype = list_val_dtype(column)

            if not col_shape.is_list or col_shape.is_unknown:
                col_shape = DefaultShapes.LIST

        return output_col_schema.with_dtype(column_dtype).with_shape(col_shape)

    # TODO: Turn this into a function
    def _validate_dtypes(self, node, output_data):
        for col_name, output_col_schema in node.output_schema.column_schemas.items():
            # Validate that the dtypes match but only if they both exist
            # (since schemas may not have all dtypes specified, especially
            # in the tests)
            output_schema_dtype = output_col_schema.dtype.without_shape
            output_data_dtype = md.dtype(output_data.dtype).without_shape
            if (
                output_schema_dtype
                and output_data_dtype
                and output_schema_dtype != md.string
                and output_schema_dtype != output_data_dtype
            ):
                raise TypeError(
                    f"Dtype discrepancy detected for column {col_name}: "
                    f"operator {node.op.label} reported dtype "
                    f"`{output_schema_dtype}` but returned dtype "
                    f"`{output_data_dtype}`."
                )

    # TODO: Turn this into a function
    def _combine_node_outputs(self, node, transformed, output):
        node_output_cols = _get_unique(node.output_schema.column_names)

        # dask needs output to be in the same order defined as meta, reorder partitions here
        # this also selects columns (handling the case of removing columns from the output using
        # "-" overload)
        if output is None:
            output = transformed[node_output_cols]
        else:
            output = concat_columns([output, transformed[node_output_cols]])

        return output


class DaskExecutor:
    """
    An executor for running Merlin operator DAGs as distributed Dask jobs
    """

    def __init__(self, client=None):
        self._executor = LocalExecutor()

        # Deprecate `client`
        if client is not None:
            set_client_deprecated(client, "DaskExecutor")

    def __getstate__(self):
        # dask client objects aren't picklable - exclude from saved representation
        return {k: v for k, v in self.__dict__.items() if k != "client"}

    def transform(
        self,
        ddf,
        graph,
        output_dtypes=None,
        additional_columns=None,
        capture_dtypes=False,
        strict=False,
    ):
        """
        Transforms all partitions of a Dask Dataframe by applying the operators
        from a collection of Nodes
        """
        nodes = []
        if isinstance(graph, Graph):
            nodes.append(graph.output_node)
        elif isinstance(graph, Node):
            nodes.append(graph)
        elif isinstance(graph, list):
            nodes = graph
        else:
            raise TypeError(
                f"DaskExecutor detected unsupported type of input for graph: {type(graph)}."
                " `graph` argument must be either a `Graph` object (preferred)"
                " or a list of `Node` objects (deprecated, but supported for backward"
                " compatibility.)"
            )

        self._clear_worker_cache()

        # Check if we are only selecting columns (no transforms).
        # If so, we should perform column selection at the ddf level.
        # Otherwise, Dask will not push the column selection into the
        # IO function.
        if not nodes:
            return ddf[_get_unique(additional_columns)] if additional_columns else ddf

        if isinstance(nodes, Node):
            nodes = [nodes]

        columns = list(flatten(wfn.output_columns.names for wfn in nodes))
        columns += additional_columns if additional_columns else []

        if isinstance(output_dtypes, dict):
            for col_name, col_dtype in output_dtypes.items():
                if col_dtype:
                    output_dtypes[col_name] = md.dtype(col_dtype).to_numpy

        if isinstance(output_dtypes, dict) and isinstance(ddf._meta, pd.DataFrame):
            dtypes = output_dtypes
            output_dtypes = type(ddf._meta)({k: [] for k in columns})
            for col_name, col_dtype in dtypes.items():
                output_dtypes[col_name] = output_dtypes[col_name].astype(col_dtype)

        elif not output_dtypes:
            # TODO: constructing meta like this loses dtype information on the ddf
            # and sets it all to 'float64'. We should propagate dtype information along
            # with column names in the columngroup graph. This currently only
            # happens during intermediate 'fit' transforms, so as long as statoperators
            # don't require dtype information on the DDF this doesn't matter all that much
            output_dtypes = type(ddf._meta)({k: [] for k in columns})

        return ensure_optimize_dataframe_graph(
            ddf=ddf.map_partitions(
                self._executor.transform,
                nodes,
                additional_columns=additional_columns,
                capture_dtypes=capture_dtypes,
                strict=strict,
                meta=output_dtypes,
                enforce_metadata=False,
            )
        )

    def fit(self, ddf, nodes, strict=False):
        """Calculates statistics for a set of nodes on the input dataframe

        Parameters
        -----------
        ddf: dask.Dataframe
            The input dataframe to calculate statistics for. If there is a
            train/test split this should be the training dataset only.
        """
        stats = []
        for node in nodes:
            # Check for additional input columns that aren't generated by parents
            addl_input_cols = set()
            if node.parents:
                upstream_output_cols = sum(
                    [upstream.output_columns for upstream in node.parents_with_dependencies],
                    ColumnSelector(),
                )
                addl_input_cols = set(node.input_columns.names) - set(upstream_output_cols.names)

            # apply transforms necessary for the inputs to the current column group, ignoring
            # the transforms from the statop itself
            transformed_ddf = self.transform(
                ddf,
                node.parents_with_dependencies,
                additional_columns=addl_input_cols,
                capture_dtypes=True,
                strict=strict,
            )

            try:
                stats.append(node.op.fit(node.input_columns, transformed_ddf))
            except Exception:
                LOG.exception("Failed to fit operator %s", node.op)
                raise

        dask_client = global_dask_client()
        if dask_client:
            results = [r.result() for r in dask_client.compute(stats)]
        else:
            results = dask.compute(stats, scheduler="synchronous")[0]

        for computed_stats, node in zip(results, nodes):
            node.op.fit_finalize(computed_stats)

    def _clear_worker_cache(self):
        # Clear worker caches to be "safe"
        dask_client = global_dask_client()
        if dask_client:
            dask_client.run(clean_worker_cache)
        else:
            clean_worker_cache()


def _get_unique(cols):
    # Need to preserve order in unique-column list
    return list({x: x for x in cols}.keys())
