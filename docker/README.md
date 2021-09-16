# Use Docker

The Dockerfile is set up to expose port 22 and make the container accessible via SSH. This allows not only to
transfer files, but also to use the Docker container as a remote debugger for PyCharm and other development environments
. However, for this to work, the public SSH key must be included in the container.

1. Generate a SSH-Key:
    ```bash
    ssh-keygen -t rsa
    ```

2. Copy the public key and rename it to`authorized_keys` and save it in a folder called `.ssh`

3. Create an archive which contains the public key:
   ```bash
   tar -czvf .ssh.tar.gz </path/to/.ssh>
   ```
   
4. Save the `Dockerfile` and the archive `ssh_keys.tar.gz` in the same folder and run the command
   ```bash
   docker build -t realscore </path/to/folder/>
   ```