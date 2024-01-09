import streamlit as st
from PIL import Image
from torch import autocast
import torch
import requests
from PIL import Image
from PIL import ImageOps
from io import BytesIO
from diffusers import StableDiffusionImg2ImgPipeline
from PIL import ImageOps

st.set_page_config(layout="wide")
def intro():
    st.write("<h1 style='text-align: center; color: #e0392d;'>Creative Image Processing and Ad Design 🤖</h1>",
             unsafe_allow_html=True)

    # Introduction
    st.header(":blue[Introduction]")
    st.write("Hello everyone, I'm Mert Eren Dagistan, and today I'm thrilled to showcase a versatile Python program "
             "that empowers users to personalize images and create stunning advertisements. Let's explore the "
             "detailed workflow and applications of this tool.")

    # Image Customization
    st.header(":blue[1. Image Customization]")

    ## User Interaction
    st.write("<h3 style='color: #d6807a;'>1.1 User Interaction</h3>", unsafe_allow_html=True)
    st.write("- **Photo Upload:** Users can start by uploading a photo of their choice.")
    st.write(
        "- **Prompt Input:** Add a personalized touch by inputting a prompt that guides the image customization process.")
    st.write(
        "- **Color Customization:** Tailor the color palette using a hexadecimal code, providing endless creative possibilities.")

    ## Image Processing
    st.write("<h3 style='color: #d6807a;'>1.2 Image Processing</h3>", unsafe_allow_html=True)
    st.write(
        "- **Stable Diffusion Technology:** Harness the power of Stable Diffusion to generate unique and visually appealing images.")
    st.write(
        "- **Prompt Integration:** The provided prompt and color choices guide the image customization, ensuring a personalized touch.")

    ## Example Application
    st.write("<h3 style='color: #d6807a;'>1.3 Example Application</h3>", unsafe_allow_html=True)
    st.write(
        "Demonstrate a use case: Customizing a coffee photo with a prompt like 'Warm and inviting atmosphere with a touch of [user-defined color].'")

    # Ad Design
    st.header(":blue[2. Ad Design]")

    ## Continued Customization
    st.write("<h3 style='color: #d6807a;'>2.1 Continued Customization</h3>", unsafe_allow_html=True)
    st.write(
        "- **User Decision:** If the user wishes to proceed, they can further customize the generated image into a captivating advertisement.")

    ## User Inputs for Ad Design
    st.write("<h3 style='color: #d6807a;'>2.2 User Inputs for Ad Design</h3>", unsafe_allow_html=True)

    st.write("- **Logo Upload:** Users can upload their logo, adding branding elements to the advertisement.")
    st.write("- **Punchline Text:** Input a compelling punchline, attracting attention and conveying the message.")
    st.write("- **Button Text:** Specify the call-to-action text for the button, encouraging user engagement.")
    st.write(
        "- **Color Customization:** Customize the colors of both the punchline text and the button for cohesive design.")

    ## Ad Design Process
    st.write("<h3 style='color: #d6807a;'>2.3 Ad Design Process</h3>", unsafe_allow_html=True)
    st.write("- **Frame Addition:** Incorporate a white frame around the image, enhancing visual appeal.")
    st.write("- **Logo Integration:** Place the user-uploaded logo strategically on the image.")
    st.write("- **Punchline Addition:** Add a punchline below the image, conveying the advertisement's message.")
    st.write("- **Button Inclusion:** Integrate a call-to-action button, encouraging viewers to take action.")

    ## Example Application
    st.write("<h3 style='color: #d6807a;'>2.4 Example Application</h3>", unsafe_allow_html=True)
    st.write("Showcase the final output: An eye-catching advertisement poster generated based on user preferences.")

    # Advantages and Use Cases
    st.header(":blue[3. Advantages and Use Cases]")

    ## Advantages
    st.write("<h3 style='color: #d6807a;'>3.1 Advantages</h3>", unsafe_allow_html=True)
    st.write(
        "- **User-Friendly:** The program is designed for ease of use, making image customization and ad design accessible to everyone.")
    st.write("- **Professional Results:** Stable Diffusion ensures professional and aesthetic outcomes.")

    ## Use Cases
    st.write("<h3 style='color: #d6807a;'>3.2 Use Cases</h3>", unsafe_allow_html=True)
    st.write("- **Product Marketing:** Enhance product photos for marketing campaigns.")
    st.write("- **Ad Campaigns:** Create tailored advertisements with specific branding elements.")
    st.write("- **Creative Projects:** Use the tool for artistic projects with customizable visuals.")

    # Conclusion
    st.header(":blue[Conclusion]")
    st.write(
        "In conclusion, this program offers a seamless experience for users to customize images and create visually striking advertisements. "
        "Its versatility, user-friendly design, and powerful capabilities make it an ideal tool for various applications.")
    st.write("Thank you for your attention!")

def generation_page():
    torch.cuda.empty_cache()

    import os

    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


    @st.cache_resource
    def load_pipeline():
        device = "cuda"
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            revision="fp16",
            torch_dtype=torch.float16,
            use_auth_token=True
        ).to(device)
        return pipe


    # Load the pipeline once
    pipe = load_pipeline()


    def user_color_adder(prompt, user_color):
        """
        color adder function according to user_color & prompt
        :param prompt: text -> string
        :param user_color: color(hex code) -> string
        :return: text -> string
        """
        text = prompt + "with a specified color hex code: (" + user_color + ") , COLOR : "
        return text


    def create_image(init_image, prompt, user_color):
        """
        creating image using prompt & init image (user's selected)
        :param init_image: Image
        :param prompt: user_prompt -> string
        :param user_color: color -> string (hex code)
        :return: images -> Image
        """
        # resizing
        init_image = init_image.resize((512, 512))  # square / can be (768,512)
        user_color_adder(prompt, user_color)  # that func add color to prompt
        prompt_last = user_color_adder(prompt, user_color)  # added color palette
        with autocast("cuda"):
            images = pipe(prompt=prompt_last, image=init_image, strength=0.75, guidance_scale=7.5)  # generating images
        return images  # return image


    def task1_imp(user_image, user_prompt, user_color):
        """
        Processes Task 1 by adding color to the user prompt text on the given image.

        :param user_image: image (from task1)
        :param user_prompt: string -> text (user_prompt)
        :param user_color: string -> color (hex code)
        :return: Image -> image & save selected path
        """
        # open with image to user_photo
        # user_image = Image.open(user_image)
        # color adder func implementation
        user_color_adder(user_prompt, user_color)
        # generation of image
        images = create_image(user_image, user_prompt, user_color)
        # saving
        if images and hasattr(images, 'images') and isinstance(images.images, list) and isinstance(images.images[0],
                                                                                                   Image.Image):

            # sonuc_path = "C:/Users/ASUS/PycharmProjects/spaceship-titanic/task1_generated_photo/task1_11.png"
            # images.images[0].save(sonuc_path)
            return images
        else:
            print("Error: The images object is empty or does not contain a PIL Image object.")


    st.header(":red[Advertisement Creation]")

    st.markdown("Task 1")

    uploaded_file = st.file_uploader("Select photo", type=["jpg", "jpeg", "png"])

    prompt = st.text_input("Enter the prompt: ")

    color = st.text_input("Enter the color (HEX): ")

    if uploaded_file is not None:
        # Seçilen dosyayı aç
        image = Image.open(uploaded_file)
        button_clicked = st.button("Generate")

        if button_clicked:
            st.write("Generating image...")
            images = task1_imp(user_image=image, user_prompt=prompt, user_color=color)
            st.image(images.images[0], caption="Seçilen Fotoğraf", use_column_width=True)

page_names_to_funcs = {
    "Homepage": intro,
    "Generate Advertisement": generation_page,
}

page_name = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
page_names_to_funcs[page_name]()