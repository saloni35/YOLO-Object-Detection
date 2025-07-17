**Building the Docker Image**
Navigate to your project's root directory (where the Dockerfile is located) in your terminal and run the following command to build the Docker image:

```
docker build -t my-object-detector .
```

Note- You can specify BUILD_TYPE arg with option "cpu" if the host os does not have NVIDIA GPU support(e.g.on MacOS)

```
docker build --build-arg BUILD_TYPE=cpu -t my-object-detector .
```

**Running the Docker Container**

```
docker run -p 8000:8000 --name my-running-detector my-object-detector
```

**Exposing the Application with ngrok**
To share your running application with others over the internet, use ngrok.

Ensure ngrok is authenticated: If you haven't already, connect ngrok to your account using your authtoken:

```
ngrok config add-authtoken <YOUR_NGROK_AUTH_TOKEN>
```

(Replace <YOUR_NGROK_AUTH_TOKEN> with the token from your ngrok dashboard.)

Start the ngrok tunnel: Open a new terminal window (keep your Docker container running in the first terminal) and run ngrok, pointing it to the port your Docker container is exposed on (8000):

```
ngrok http 8000
```

Share the Public URL:
Ngrok will display a public https:// URL in your terminal (e.g., https://abcdef12345.ngrok-free.app).
