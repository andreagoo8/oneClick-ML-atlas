# How to use oneClick-ML on the Atlas cluster with `dhcli`

This guide translates `main.ipynb` into terminal commands: create or retrieve the
project, upload the CSV and configuration, register `main:helper_main`, build its
image, and submit a training job. It keeps the notebook's names, Python 3.10, and
resource settings. Run the steps in order in the same Bash terminal, from the
repository root (the directory containing `main.ipynb`). The commands submit work
to Digitalhub; training happens on the cluster.

## 1. Prepare the CLI and the input files

Install a `dhcli` release compatible with Digitalhub 0.15 for your operating system
and architecture, extract it, and put the executable on `PATH`. See the official
[installation instructions](https://scc-digitalhub.github.io/docs/0.15/cli/installation/)
and [release downloads](https://github.com/scc-digitalhub/digitalhub-cli/releases).
The local tools used below are Bash, Python 3 (only its standard library), and
`jq`. The ML dependencies are installed in the cluster image during the build.

```bash
dhcli --version
python3 --version
jq --version
mkdir -p dhcli
export PROJECT_NAME='oneclick-ml-atlas'
```

Prepare `oneClick-ML/data.csv` and edit `oneClick-ML/config.ini` before uploading:

- Set `y_label` to the target column in your CSV and review `col_to_drop`,
  classifiers, cross-validation, and other analysis settings.
- Keep the CSV filename `data.csv` and `input_path = local_data/data/data.csv`.
  The helper downloads the CSV into `local_data/data`.
- Keep `output_path` beneath `local_data/results/`, for example the existing
  `local_data/results/no_active_at_bl/`. The helper uploads that results tree.
- The supplied `path_to_feature_types = ../data/clusters/feature_list_types`
  points outside this repository. If you use feature lists, copy their CSVs into
  `oneClick-ML/feature_list_types/` and set
  `path_to_feature_types = feature_list_types`. Expected names are
  `nominal_features.csv`, `numerical_features.csv`, `ordinal_features.csv`, and
  `binary_features.csv`. These files must travel with the source; workspace paths
  are not automatically available in a job. The loader warns and applies its
  fallback when lists are absent, so check that this behavior suits your data.

`params.yaml` is also required: `main.py` opens it using a relative path. The
archive prepared in step 4 puts it next to `main.py` and `lib/`, at its root.

## 2. Connect to Digitalhub and create or retrieve the project

Replace the endpoint placeholder with the **Core API base URL** for Atlas,
available from your platform administrator. Register it once, select the
environment, then authenticate. Use the configured environment directly if it
already exists. This follows the documented
[CLI connection flow](https://scc-digitalhub.github.io/docs/0.15/cli/usage/).

```bash
export ATLAS_CORE_ENDPOINT='https://<your-atlas-core-endpoint>'
dhcli register -e atlas "$ATLAS_CORE_ENDPOINT"
dhcli use atlas
dhcli login -e atlas
dhcli list-env
```

The CLI stores its connection configuration in `.dhcore.ini`. Uploads and
downloads also require access to the platform's S3-compatible storage, including
the correct endpoint and credentials for your installation. If Core access works
but transfer fails, check the environment's storage configuration with your
administrator. Do not copy workspace-only filesystem paths into remote sources.

Retrieve the project, corresponding to `dh.get_or_create_project(...)`:

```bash
dhcli get projects "$PROJECT_NAME"
```

If the project does not exist, create it, then retrieve it again. If retrieval
fails because of authentication, connectivity, or permissions, resolve that
error first.

```bash
dhcli create projects -n "$PROJECT_NAME"
dhcli get projects "$PROJECT_NAME"
```

Use the exact project name shown in Core if reusing one: the notebook uses
`oneclick-ml-atlas`, while the README uses different capitalization.

## 3. Load or retrieve the input dataset and configuration

An artifact combines a catalog entry with stored files. These uploads replace
the notebook's two `project.log_artifact(...)` calls. The transfer command
uploads the bytes as well as registering the artifact; creating metadata alone
would not make a local file accessible to the cluster. See
[artifact transfers](https://scc-digitalhub.github.io/docs/0.15/cli/usage/#uploaddownload-a-filefolder).

Run these commands for new inputs:

```bash
dhcli upload -p "$PROJECT_NAME" artifact \
  -n csv_input_data -f oneClick-ML/data.csv
dhcli upload -p "$PROJECT_NAME" artifact \
  -n config_ini_file -f oneClick-ML/config.ini
```

If these inputs are already registered and you want to reuse them, skip the
uploads. In either case, fetch their metadata, corresponding to
`project.get_artifact(...)`:

```bash
dhcli get -p "$PROJECT_NAME" -o json artifacts -n csv_input_data \
  > dhcli/csv-artifact.json
dhcli get -p "$PROJECT_NAME" -o json artifacts -n config_ini_file \
  > dhcli/config-artifact.json
jq '{id, name, state: .status.state, path: .spec.path}' dhcli/csv-artifact.json
jq '{id, name, state: .status.state, path: .spec.path}' dhcli/config-artifact.json
```

Both artifacts must be `READY`. Record their version-specific entity keys, the
equivalent of `csv_data.key` and `config_data.key` in the notebook:

```bash
export CSV_KEY="$(jq -er '"\(.kind)://\(.project)/\(.name):\(.id)"' dhcli/csv-artifact.json)"
export CONFIG_KEY="$(jq -er '"\(.kind)://\(.project)/\(.name):\(.id)"' dhcli/config-artifact.json)"
printf '%s\n' "$CSV_KEY" "$CONFIG_KEY"
```

Keys have the form `artifact://oneclick-ml-atlas/csv_input_data:<id>`. They identify
catalog objects; S3 paths identify stored files. The job's `inputs` need entity
keys. To select an older version, list versions and replace `-n name` above with
its ID, then recalculate the keys:

```bash
dhcli list -p "$PROJECT_NAME" artifacts -n csv_input_data
dhcli list -p "$PROJECT_NAME" artifacts -n config_ini_file
```

## 4. Understand, package, and declare the main helper function

A Digitalhub function describes reusable executable code. Registration, image
building, and execution are separate operations. Here the registered handler is
the existing `helper_main` in `oneClick-ML/main.py`. At runtime it receives the
project automatically, downloads the two input artifacts, calls the original ML
program, and logs `local_data/results` as `output_data`.

The notebook's SDK call prepares its local `code_src` directory for remote use.
For the CLI, explicitly package the local source and upload it. Digitalhub
supports [S3 ZIP sources](https://scc-digitalhub.github.io/sdk-docs/0.15/reference/configuration/code_src/s3/)
using `zip+s3://...` and a `module:callable` handler. This captures local edits
without requiring a Git push.

Create a ZIP with `main.py`, `params.yaml`, `requirements.txt`, `lib/`, and any
feature lists at the archive root. Inputs remain separate artifacts:

```bash
python3 - <<'PY'
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

root = Path('oneClick-ML')
files = [root / name for name in ('main.py', 'params.yaml', 'requirements.txt')]
files += sorted((root / 'lib').rglob('*.py'))
files += sorted((root / 'feature_list_types').rglob('*.csv'))
with ZipFile('dhcli/oneclick-source.zip', 'w', ZIP_DEFLATED) as archive:
    for path in files:
        archive.write(path, path.relative_to(root))
PY

dhcli upload -p "$PROJECT_NAME" artifact \
  -n oneclick_source -f dhcli/oneclick-source.zip
dhcli get -p "$PROJECT_NAME" -o json artifacts -n oneclick_source \
  > dhcli/source-artifact.json
export SOURCE_PATH="$(jq -er '.spec.path' dhcli/source-artifact.json)"
```

Check that this source artifact is `READY` and its path is the uploaded ZIP's
`s3://.../oneclick-source.zip` address:

```bash
jq '{id, state: .status.state, path: .spec.path}' dhcli/source-artifact.json
```

Generate `dhcli/main-function.yaml`. JSON is valid YAML, so this uses Python's
standard library to serialize the definition and all dependency pins without
requiring a local Digitalhub SDK installation. Unlike the SDK's convenience
argument, the CLI definition contains the actual requirements list rather than
the path to a local requirements file.

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

source_path = os.environ['SOURCE_PATH']
if not source_path.startswith('s3://') or not source_path.endswith('.zip'):
    raise ValueError('SOURCE_PATH must point to the uploaded S3 ZIP file')
requirements = [
    line.strip()
    for line in Path('oneClick-ML/requirements.txt').read_text().splitlines()
    if line.strip() and not line.lstrip().startswith('#')
]
definition = {
    'kind': 'python',
    'name': 'main_function',
    'project': os.environ['PROJECT_NAME'],
    'spec': {
        'python_version': 'PYTHON3_10',
        'requirements': requirements,
        'source': {
            'lang': 'python',
            'source': 'zip+' + source_path,
            'handler': 'main:helper_main',
        },
    },
}
Path('dhcli/main-function.yaml').write_text(json.dumps(definition, indent=2) + '\n')
PY

dhcli create -p "$PROJECT_NAME" functions -f dhcli/main-function.yaml
dhcli get -p "$PROJECT_NAME" -o json functions -n main_function \
  > dhcli/main-function.json
export FUNCTION_ID="$(jq -er '.id' dhcli/main-function.json)"
```

The nested `spec.source.source` and `spec.source.handler` fields are the serialized
form of the SDK's `code_src` and `handler` arguments; see the version 0.15
[source schema](https://github.com/scc-digitalhub/digitalhub-sdk-runtime-python/blob/0.15.0/digitalhub_runtime_python/entities/function/python/models.py).
The function retains every dependency in the repository's requirements file.

If the function already exists and its source and dependencies are suitable,
skip packaging and creation: run only the `get` and `FUNCTION_ID` commands above.
Use a specific ID instead of `-n main_function` to retrieve a known version.
Pinning the ID also ensures the build and job use the same function version.

## 5. Build the main helper function

Create `dhcli/build.yaml` with the notebook's build resources:

```bash
cat > dhcli/build.yaml <<'YAML'
spec:
  resources:
    cpu: "4"
    mem: "8Gi"
    disk: "10Gi"
YAML

dhcli run -p "$PROJECT_NAME" -i "$FUNCTION_ID" \
  -f dhcli/build.yaml python+build
dhcli list -p "$PROJECT_NAME" runs -k 'python+build:run'
```

`dhcli run` submits asynchronously. Unlike the notebook's `wait=True`, command
success only confirms submission. Select the build run belonging to this function
in the run list or Core UI, copy its ID, and inspect it:

```bash
export BUILD_RUN_ID='<build-run-id>'
dhcli get -p "$PROJECT_NAME" -o yaml runs "$BUILD_RUN_ID"
dhcli log -p "$PROJECT_NAME" -f "$BUILD_RUN_ID"
```

Repeat `get` until `status.state` is `COMPLETED`. Do not submit training if the
build fails; inspect the logs and fix the dependency or build error first. Logs
alone do not replace the state check. These commands follow the documented
[run monitoring flow](https://scc-digitalhub.github.io/docs/0.15/cli/usage/#monitor-a-run).

After completion, fetch the function again to get the image reference, replacing
`main_func.refresh().spec.image`:

```bash
dhcli get -p "$PROJECT_NAME" -o json functions "$FUNCTION_ID" \
  > dhcli/main-function.json
export FUNCTION_IMAGE="$(jq -er '.spec.image | select(length > 0)' dhcli/main-function.json)"
printf '%s\n' "$FUNCTION_IMAGE"
```

The [Python build operation](https://scc-digitalhub.github.io/docs/0.15/runtimes/python/)
prepares an image for subsequent executions. Proceed only once the image
reference is populated. A previously built function can start directly at step 6
after retrieving its ID and checking its image.

## 6. Run the main helper function on the cluster

Generate `dhcli/job.yaml` with the input keys recorded in step 3:

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

definition = {
    'spec': {
        'inputs': {
            'csv_file': os.environ['CSV_KEY'],
            'config_file': os.environ['CONFIG_KEY'],
        },
        'parameters': {'output_artifact_name': 'output_data'},
        'resources': {'cpu': '4', 'mem': '8Gi', 'disk': '20Gi'},
        'local_execution': False,
    },
}
Path('dhcli/job.yaml').write_text(json.dumps(definition, indent=2) + '\n')
PY

dhcli run -p "$PROJECT_NAME" -i "$FUNCTION_ID" \
  -f dhcli/job.yaml python+job
dhcli list -p "$PROJECT_NAME" runs -k 'python+job:run'
```

The argument names `csv_file`, `config_file`, and `output_artifact_name` must match
`helper_main`. Artifact references belong in `inputs`; the output name belongs in
`parameters`. `project` is injected by the runtime. See the
[Python job reference](https://scc-digitalhub.github.io/sdk-docs/0.15/reference/runtimes/python/python/python-job/).

Keep the outer `spec` wrapper in both run files: the
[0.15 CLI implementation](https://github.com/scc-digitalhub/digitalhub-cli/blob/0.15.2/handlers/adapter/run.go)
reads additional run settings from that mapping. The CLI resolves function and
task references and submits a remote run, so no separate task creation or local
execution environment is needed.

Copy the training run ID from the list or Core UI and monitor it:

```bash
export JOB_RUN_ID='<training-run-id>'
dhcli get -p "$PROJECT_NAME" -o yaml runs "$JOB_RUN_ID"
dhcli log -p "$PROJECT_NAME" -f "$JOB_RUN_ID"
```

The run also appears in Core's **Runs** section. Wait for `COMPLETED` and inspect
the logs for training errors. The existing helper catches exceptions and prints
`Error during training: ...` without re-raising them, so a completed run alone
does not prove training succeeded. Verify a newly produced output artifact too.

## 7. Inspect and download results

The helper explicitly logs an **artifact**, not a Digitalhub Model entity. Find
`output_data` under **Artifacts** in Core, or use:

```bash
dhcli list -p "$PROJECT_NAME" artifacts -n output_data
dhcli get -p "$PROJECT_NAME" -o json artifacts -n output_data \
  > dhcli/output-artifact.json
jq '{id, name, state: .status.state, metadata, files: .status.files}' \
  dhcli/output-artifact.json
export OUTPUT_ID="$(jq -er '.id' dhcli/output-artifact.json)"
dhcli download -p "$PROJECT_NAME" artifact "$OUTPUT_ID" \
  -d results/output_data
```

Check that the artifact is `READY`, contains the expected results, and was created
by the current training run (check the creation time and run relationship in
Core/metadata). An older `output_data` can remain after a failed training run.
When jobs share an output name, choose the correct artifact version explicitly
from the list rather than assuming the latest one belongs to your run.

## 8. Reuse the pipeline or update its code

- **New terminal session:** reconnect/select `atlas`, set `PROJECT_NAME`, fetch
  the two artifacts and calculate their keys in step 3, then retrieve the function
  and `FUNCTION_ID` in step 4. If its image exists, continue with step 6.
- **Different data or configuration:** upload the changed files in step 3,
  retrieve the intended versions, recalculate their keys, and regenerate
  `dhcli/job.yaml`. Change `output_artifact_name` there to distinguish experiments.
  Reuse the built function when code and dependencies are unchanged.
- **Changed source:** repeat packaging and source upload in step 4, regenerate
  the definition, and create a new function version. Retrieve its new ID, build
  that version, and submit the job with that same ID. Rebuild whenever dependencies
  change. This is also a straightforward option for code-only changes.

For the notebook's code-only image-reuse option, first retrieve the existing
function's image as in step 5. After generating the new definition with the new
source, add that image before creating the new version:

```bash
python3 - <<'PY'
import json
import os
from pathlib import Path

path = Path('dhcli/main-function.yaml')
definition = json.loads(path.read_text())
definition['spec']['image'] = os.environ['FUNCTION_IMAGE']
path.write_text(json.dumps(definition, indent=2) + '\n')
PY
```

Then execute the function creation, retrieval, and `FUNCTION_ID` assignment from
step 4 and proceed to step 6. Use this shortcut only when the existing image
already contains all required dependencies; otherwise build the new version.

The CLI's create/get/list/run/log/download syntax is documented in the
[Digitalhub 0.15 command reference](https://scc-digitalhub.github.io/docs/0.15/cli/commands/).
