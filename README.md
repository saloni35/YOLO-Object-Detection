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

**Running API Service/Training Model/Replace Trained Model**

***Open an interactive shell inside your running container so you can run commands manually***
```
docker exec -it  my-running-detector bash
```

**Now execute any of the following commands as needed**

***Start training script***
Before starting training, replace  dataset in YOLO format by copying into dataset directory-

```
docker cp /path/on/host my-running-detector:/app/dataset
```

Now start the training

```
python3 train.py
```

***Replace your trained model to be used by API server***
```
cp saved_model/best.pt custom_model/best.pt
```

***Restart API server which serves object detection***
```
pkill -f "uvicorn main:app"
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
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
