# Save schema files as JSON in `.merlin` directory

* **Status:** pending
* **Last Updated:** 2022-12
* **Objective:** Set a default file type and directory structure for schema information.

## Context & Problem Statement

Merlin Schemas are used throughout the Merlin ecosystem. Finding the schema information from any Merlin artifact should be standardized to make it easy for other parts of Merlin to consume that data. The data should be in a standardized structure and format so that all Merlin components can easily read and consume the merlin schema information produced by an artifact.

## Priorities & Constraints <!-- optional -->

* We'd like the schema files to human readable and editable
* We still want validation of the contents when saving and loading

## Considered Options

* Option 1: .pbtxt
* Option 2: .json

## Decision

Chosen option: JSON, but continuing to use a Protobuf schema to validate the contents. It's already in use by the Merlin Models library, and is the most readable and hand-editable of the formats supported by Better Proto. JSON schema files should be stored in a `.merlin` sub-directory inside the directory of other exported artifacts (e.g. saved models, Parquet files.)

### Expected Consequences <!-- optional -->

* The `merlin.io.Dataset` class in Merlin Core will need to change from exporting .pbtxt to exporting .json (issue link goes here)