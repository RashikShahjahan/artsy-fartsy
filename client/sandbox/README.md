The image is built from the repository root. `Dockerfile.dockerignore` overrides
the root renderer-only ignore rules. Keep this directory's Python source inside
the image; the frontend deployment only needs the TypeScript orchestration.

`agent.py` runs the existing mini agent with the native Tinker adapter. It never
executes generated shell commands. `action.py` executes one action in a separate
VM and uses the existing submission/preview protocol. `snapshot.py` saves partial
work on turn exhaustion. The renderer VM has no network access or credentials.

To prepare a snapshot without uploading an OCI image, run these from the
repository root with the Vercel CLI authenticated. Choose a fresh build name:

```sh
python3 client/sandbox/package.py /tmp/artsy-runtime.tar.gz
vercel sandbox create --name artsy-runtime-build --image vercel/sandbox/python:3.14 --timeout 20m --snapshot-expiration 30d --keep-last-snapshots 1 --project artsy
vercel sandbox copy /tmp/artsy-runtime.tar.gz artsy-runtime-build:/vercel/sandbox/runtime.tar.gz --project artsy
vercel sandbox copy client/sandbox/prepare.sh artsy-runtime-build:/vercel/sandbox/prepare.sh --project artsy
vercel sandbox exec --sudo --timeout 12m --project artsy artsy-runtime-build -- bash /vercel/sandbox/prepare.sh
vercel sandbox snapshot artsy-runtime-build --stop --expiration 30d --project artsy
```

Set `ARTSY_SANDBOX_SNAPSHOT` to the returned snapshot ID and deploy the client.
No inference credentials enter the build sandbox or its snapshot. Job creation
clones it into two separate VMs and injects Tinker credentials only into the
trusted agent VM. Snapshots expire after 30 days without use. Refresh the runtime snapshot if it
expires. The OCI
Dockerfile remains available as an alternative runtime source.
