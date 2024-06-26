# Interview with AI
This repository contains the source code for the project "Interview with AI". It allows user to give as many mock interviews with AI for any job profile and any organisation. AI will assess the response on multiple metrics and provide a feedback on the scale 1-10 with the improvements.

This is both voice activated and user can type the source code for any algorithmic type question, It is usign GPT-4 as main judge model and Whisper for Text to Speech and Speech to Text.

## Demo (Click on the image)

[<img src="https://raw.githubusercontent.com/eshassy/Interview-with-AI/main/banner.png" width="50%">](https://youtu.be/nVvWuQhQC7g)

## Usage

- Install the required dependencies by running
```bash
pip install -r requirements.txt
```
- Get the OpenAI api key and put it into the `.env` folder
- Execute the `main` file as
```bash
gradio main.py
```
