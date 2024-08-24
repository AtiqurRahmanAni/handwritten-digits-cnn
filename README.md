A simple webapp to recognize handwritten digits from 0 to 9 and visualize activation maps

![Screenshot 2024-08-24 194026](https://github.com/user-attachments/assets/1aac55e0-8081-4737-bfe2-5363fc5c2da3)

## Project requirements
Node version **20** or higher is required for the frontend, and Python version **3.10** or higher 
is required for the backend to run this project. The project could also work on a smaller version.

### How to run frontend
- Go to the frontend directory
- create `.env` file
- Copy everything from the `.env.example` file
- Assign values
- Run `npm i` and then `npm run dev`
- To run this project in production mode, first build the project using `npm run build`, and then use `npm run preview` to run.

There is a `VITE_API_BASE_URL` variable in the `.env` file. The value of that variable will be the base url of the backend. 
For example, if the backend runs on `http://localhost:8000`, the value of the `VITE_API_BASE_URL` variable will be `http://localhost:8000`.

### How to run backend
- Go to the backend directory
- create `.env` file
- Copy everything from the `.env.example` file
- Assign values
- run `.\venv\Scripts\activate` to create a new virtual environment
- Run `pip install -r requirements.txt` to install all the dependencies
- Run `fastapi dev` for run
There is an `ALLOWED_ORIGIN` variable in the `.env` file. The value of this variable will be the url of the frontend. For instance,
if the frontend runs on `http://localhost:5173`, the value of the variable will be this url.

Project demo: [Link](https://www.youtube.com/watch?v=FFNXYUJcOMA)

Live: [Link](https://handwritten-digits-cnn-uopk.vercel.app/)
