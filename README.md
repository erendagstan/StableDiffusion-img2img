This project implements an image generation pipeline using the Stable Diffusion model for tasks like adding text and creating banners.

## Table of Contents

- [Task 1](#task-1)
- [Task 2](#task-2)
- [Task 3](#task-3)
- [Usage](#usage)
- [License](#license)
- [App](#app)

## Task 1

Task 1 involves processing an image by adding colored text based on user input.

## Task 2

Task 2 extends the functionality by adding a frame, logo, punchline text, and a button to the processed image.

## Task 3

Task 3 represents the Flask application serving the image generation API.

## Usage
1. Install the required dependencies:

   ```bash
   pip install -r requirements.txt

## License
This project is licensed under the MIT License. See the [LICENSE.md](LICENSE) file for details.

## App
Selected user image :

<img src="https://github.com/erendagstan/StableDiffusion-img2img/assets/86521359/b88198f7-e078-4733-803a-f213600f91d2" width=300 height=300>

### Task 1 

  ```python
  Resim dosyasının yolunu girin (örneğin: image.png): >? C:\Users\ASUS\PycharmProjects\spaceship-titanic\photos\kahve2.png
  Promptu girin: >? A coffee photo with heart-shaped patterns, creating a warm atmosphere, featuring rising steam above the coffee, and highlighted by delightful foam.
  Görselde kullanmak için bir renk girin (hex kodu): >? #000000
  ```
Generated image after prompt & color :

<img src="https://github.com/erendagstan/StableDiffusion-img2img/assets/86521359/c8e58673-8f5f-49c6-9724-3b6d2165c42a">


### Task 2

  ```python
  Logo dosyasının yolunu girin (örneğin: logo.png): >? C:\Users\ASUS\PycharmProjects\spaceship-titanic\logos\cland.png
  Button & Punchline için bir renk girin (hex kodu): >? #000000
  Punchline için bir text girin (Alt satıra geçmek için '\n' kullanabilirsiniz): >? Come, Enjoy\n&Drink Coffee!
  Button için bir text girin: >? Buy a coffee! ->
  ```
Generated image after punchline, logo, button, button & punchline color :

<img src="https://github.com/erendagstan/StableDiffusion-img2img/assets/86521359/253b2d57-7ae1-400a-8ef9-646f7864915e">
