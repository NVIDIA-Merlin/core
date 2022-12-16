# Create a set of Merlin dtypes to use in schemas

* **Status:** pending
* **Last Updated:** 2022-12
* **Objective:** Set a standard for dtypes in Merlin schemas

## Context & Problem Statement

The Merlin ecosystem interacts with many different packages like, numpy, cupy, torch, tensorflow, feast, pandas and cudf. These packages have dtypes, some are similar but others are unique. Standardizing dtypes in Merlin would reduce friction when moving between merlin components that may leverage different packages. Using Merlin-specific dtypes would allow us to create a bridge between the dtypes from other packages. We're currently using Numpy dtypes as the standard, but they aren't comprehensive enough to capture dtype information from the other frameworks we use.

## Priorities & Constraints <!-- optional -->
* We want to avoid losing type information to the Numpy `object` type when other frameworks use a more specific type
* We want to support dtypes that are not handled by Numpy but are important for Merlin (i.e. cudf's List dtype)
* We want to be able to call out unknown dtypes explicitly, which Numpy doesn't have a way to do (other than `object`)
* We don't want to change the API too far away from the Numpy dtypes, since they're already familiar

## Considered Options

* Option 1: Create new Merlin dtypes
* Option 2: Stick with Numpy dtypes and apply workarounds

## Decision

Chosen option: Create new Merlin dtypes. We'll hew close to the Numpy dtype API, and deviate only where it allows us add functionality that Numpy doesn't have or simplify the implementation to remove complexity that we don't need.

### Expected Consequences <!-- optional -->

* Most Merlin libraries will need at least minor updates since the new dtypes will be a breaking change
* We'll be able to get rid of a lot of dtype translation tables that are hardcoded in various places throughout the Merlin code base
