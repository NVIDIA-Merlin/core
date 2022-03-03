# Documentation

This folder contains the scripts necessary to build documentation for the
merlin-core library. You can find the [generated documentation
here](https://nvidia-merlin.github.io/core).



1. Follow the instructions to create a Python developer environment. See the
   [installation instructions](https://github.com/NVIDIA-Merlin/models).

2. Install required documentation tools and extensions:

   ```sh
   cd models
   pip install -r requirements/dev.txt
   ```

3. Navigate to `models/docs/` and transform the documentation to HTML output:

   ```sh
   make html
   ```

   This should run Sphinx in your shell, and output HTML in
   `build/html/index.html`

## Preview the documentation build

1. To view the docs build, run the following command from the `build/html`
   directory:

   ```sh
   http-server build/html
   ```

   (http-server can be installed with `npm install --global http-server`)
