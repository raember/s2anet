# Use Docker

The Dockerfile-dev is set up to expose port 22 and make the container accessible via SSH. This allows not only to
transfer files, but also to use the Docker container as a remote debugger for PyCharm and other development environments
. However, for this to work, the public SSH key must be included in the container.

1. Generate a SSH-Key:
    ```bash
    ssh-keygen -t rsa
    ```

2. Copy the public key and rename it to`authorized_keys`.
   ```bash
   mv <path/to/>id_rsa.pub authorized_keys
   ```
   
4. Save the `Dockerfile` and the archive `.ssh.tar.gz` in the same folder and run the command
   ```bash
   ./docker/build.sh dev
   ```

## TLDR:

- Base image: `./docker/build.sh`
- Prod-image: `./docker/build.sh prod`
- Dev-image: Put public key as `authorized_keys` in project root folder, then `./docker/build.sh dev`
