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
import warnings
from enum import Enum
from typing import List, Set, Union


class Tags(Enum):
    """Standard tags used in the Merlin ecosystem"""

    # Feature types
    CATEGORICAL = "categorical"
    CONTINUOUS = "continuous"
    LIST = "list"
    SEQUENCE = "sequence"
    TEXT = "text"
    TOKENIZED = "tokenized"
    TIME = "time"

    # Feature context
    ID = "id"
    USER = "user"
    ITEM = "item"
    SESSION = "session"
    CONTEXT = "context"

    # Target related
    TARGET = "target"
    REGRESSION = "regression"
    CLASSIFICATION = "classification"

    BINARY = "binary"
    MULTI_CLASS = "multi_class"

    # Deprecated compound tags
    USER_ID = "user_id"
    ITEM_ID = "item_id"
    SESSION_ID = "session_id"
    TEXT_TOKENIZED = "text_tokenized"
    BINARY_CLASSIFICATION = "binary_classification"
    MULTI_CLASS_CLASSIFICATION = "multi_class_classification"


TAG_COLLISIONS = {
    Tags.CATEGORICAL: [Tags.CONTINUOUS],
    Tags.CONTINUOUS: [Tags.CATEGORICAL],
}

COMPOUND_TAGS = {
    Tags.USER_ID: [Tags.USER, Tags.ID],
    Tags.ITEM_ID: [Tags.ITEM, Tags.ID],
    Tags.SESSION_ID: [Tags.SESSION, Tags.ID],
    Tags.TEXT_TOKENIZED: [Tags.TEXT, Tags.TOKENIZED],
}


class TagSet:
    """Collection that normalizes tags and prevents collisions between incompatible tags"""

    def __init__(self, tags: List[Union[str, Tags]] = None):
        if isinstance(tags, TagSet):
            tags = list(tags._tags)
        elif tags is None:
            tags = []

        self._tags: Set[Tags] = self._normalize_tags(tags)

        collisions = self._detect_collisions(self._tags, self._tags)
        if collisions:
            raise ValueError(
                f"Could not create a TagSet with the tags {self._tags}. "
                f"The following tags are incompatible: {collisions}"
            )

    def override(self, tags: List[Union[str, Tags]]) -> "TagSet":
        """Add new tags to the collection, removing any existing tags that are incompatible

        Parameters
        ----------
        tags : List[Union[str, Tags]] :
            Tags to add (and remove incompatibilities with)

        Returns
        -------
        TagSet
            A new combined set of tags with incompatible tags removed

        """
        tags = self._convert_to_tagset(tags)
        to_remove = self._detect_collisions(self._tags, tags)
        return TagSet(self - to_remove + tags)

    def __iter__(self):
        for tag in self._tags:
            yield tag

    def __len__(self):
        return len(self._tags)

    def __add__(self, tags):
        tags = self._convert_to_tagset(tags)
        return TagSet(self._tags.union(tags._tags))

    def __sub__(self, tags):
        tags = self._convert_to_tagset(tags)
        return TagSet(self._tags - tags._tags)

    def __eq__(self, tags):
        return self._tags == tags._tags

    def _detect_collisions(self, tags_a, tags_b):
        collisions = []
        for tag in tags_b:
            conflicting = TAG_COLLISIONS.get(tag, [])
            for conflict in conflicting:
                if conflict in tags_a:
                    collisions.append(conflict)
        return set(collisions)

    def _convert_to_tagset(self, tags):
        if not isinstance(tags, (list, set, TagSet)):
            tags = [tags]
        if not isinstance(tags, TagSet):
            tags = TagSet(tags)

        return tags

    def _normalize_tags(self, tags) -> Set[Tags]:
        tag_set = set(Tags[tag.upper()] if tag in Tags._value2member_map_ else tag for tag in tags)
        atomized_tags = set()

        for tag in tag_set:
            atomized_tags.add(tag)
            if tag in COMPOUND_TAGS.keys():
                warnings.warn(
                    f"Compound tags like {tag} have been deprecated "
                    "and will be removed in a future version. "
                    f"Please use the atomic versions of these tags, like {COMPOUND_TAGS[tag]}."
                )
                atomized_tags.update(COMPOUND_TAGS[tag])

        return atomized_tags

    def __repr__(self) -> str:
        return str(self._tags)


TagsType = Union[TagSet, List[str], List[Tags], List[Union[Tags, str]]]
