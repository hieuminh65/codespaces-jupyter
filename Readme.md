# AiMee AI 
## Description
AIMEE (Artificial Intelligence Mindfulness Enhancement Exercises App) will provide a library of mindfulness exercises, music, and videos to enhance emotional self-recognition and promote mindfulness. AIMEE analyzes and responds to speech input to provide recommendations on exercises.

## App onboarding
- [Google Play Store](https://play.google.com/store/apps/details?id=com.resilience.aimee)
- [Apple App Store](https://apps.apple.com/tt/app/my-aimee/id1540096035)
  
## AI model repository
This repository contains the AI model for the AiMee app. The model is trained on the [RAVDESS](https://zenodo.org/record/1188976#.YDZ6Z2hKjIU) dataset. The model is trained to classify the audio input into one of the following emotions: happy, sad, angry, fearful, and neutral.

- The dataset is available in the [data](data) folder.
- The model is available in aimee_model.ipynb notebook.
- The Flask app is available in the Run_File.py file. (The app is also deployed on AWS EC2 instance)

## Usage
1. Install dependencies
```bash
pip install -r requirements.txt
```

2. Run the backend (Flask app) 
```bash
python3 Run_File.py
```

3. Download [Postman](https://www.postman.com)

4. Send a **POST** request. The body is a form data with audio key and the type of the value is file. 
   
Check out the postman image for visual representation.

5. Send to API : {pc ip address}/process_data

Your respond should be a string of the emotion.

6. Link API on EC2 instance:
http://ec2-3-15-240-87.us-east-2.compute.amazonaws.com:8000/process_data

## Contributing
Fork or Clone the project and made changes to the code. Then create a pull request. New ideas and suggestions are always welcome.
## Privacy Policy
The code is not available for public use. The code is only available for the AiMee app. Do not use the code for any other purpose.
