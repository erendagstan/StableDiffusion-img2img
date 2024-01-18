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
from PIL import Image, ImageDraw, ImageFont

st.set_page_config(layout="wide")


def intro():
    st.write("<h1 style='text-align: center; color: #D75850;'>🤖 Creative Image Processing and Ad Design 🤖</h1>",
             unsafe_allow_html=True)

    # Introduction
    st.header(":blue[Introduction]")
    st.write("Hello everyone, I'm Mert Eren Dagistan, and today I'm thrilled to showcase a versatile Python program "
             "that empowers users to personalize images and create stunning advertisements. Let's explore the "
             "detailed workflow and applications of this tool.")

    # Image Customization
    st.header(":blue[1. Image Customization]")

    ## User Interaction
    st.write("<h3 style='color: #C7D64F;'>1.1 User Interaction</h3>", unsafe_allow_html=True)
    st.write("- **Photo Upload:** Users can start by uploading a photo of their choice.")
    st.write(
        "- **Prompt Input:** Add a personalized touch by inputting a prompt that guides the image customization process.")
    st.write(
        "- **Color Customization:** Tailor the color palette using a hexadecimal code, providing endless creative possibilities.")

    ## Image Processing
    st.write("<h3 style='color: #C7D64F;'>1.2 Image Processing</h3>", unsafe_allow_html=True)
    st.write(
        "- **Stable Diffusion Technology:** Harness the power of Stable Diffusion to generate unique and visually appealing images.")
    st.write(
        "- **Prompt Integration:** The provided prompt and color choices guide the image customization, ensuring a personalized touch.")

    ## Example Application
    st.write("<h3 style='color: #C7D64F;'>1.3 Example Application</h3>", unsafe_allow_html=True)
    st.write(
        "Demonstrate a use case: Customizing a coffee photo with a prompt like 'Warm and inviting atmosphere with a touch of [user-defined color].'")

    # Ad Design
    st.header(":blue[2. Ad Design]")

    # Continued Customization
    st.write("<h3 style='color: #C7D64F;'>2.1 Continued Customization</h3>", unsafe_allow_html=True)
    st.write(
        "- **User Decision:** If the user wishes to proceed, they can further customize the generated image into a captivating advertisement.")

    # User Inputs for Ad Design
    st.write("<h3 style='color: #C7D64F;'>2.2 User Inputs for Ad Design</h3>", unsafe_allow_html=True)

    st.write("- **Logo Upload:** Users can upload their logo, adding branding elements to the advertisement.")
    st.write("- **Punchline Text:** Input a compelling punchline, attracting attention and conveying the message.")
    st.write("- **Button Text:** Specify the call-to-action text for the button, encouraging user engagement.")
    st.write(
        "- **Color Customization:** Customize the colors of both the punchline text and the button for cohesive design.")

    # Ad Design Process
    st.write("<h3 style='color: #C7D64F;'>2.3 Ad Design Process</h3>", unsafe_allow_html=True)
    st.write("- **Frame Addition:** Incorporate a white frame around the image, enhancing visual appeal.")
    st.write("- **Logo Integration:** Place the user-uploaded logo strategically on the image.")
    st.write("- **Punchline Addition:** Add a punchline below the image, conveying the advertisement's message.")
    st.write("- **Button Inclusion:** Integrate a call-to-action button, encouraging viewers to take action.")

    # Example Application
    st.write("<h3 style='color: #C7D64F;'>2.4 Example Application</h3>", unsafe_allow_html=True)
    st.write("Showcase the final output: An eye-catching advertisement poster generated based on user preferences.")

    # Advantages and Use Cases
    st.header(":blue[3. Advantages and Use Cases]")

    # Advantages
    st.write("<h3 style='color: #C7D64F;'>3.1 Advantages</h3>", unsafe_allow_html=True)
    st.write(
        "- **User-Friendly:** The program is designed for ease of use, making image customization and ad design accessible to everyone.")
    st.write("- **Professional Results:** Stable Diffusion ensures professional and aesthetic outcomes.")

    # Use Cases
    st.write("<h3 style='color: #C7D64F;'>3.2 Use Cases</h3>", unsafe_allow_html=True)
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

    if 'generated_image' not in st.session_state:
        st.session_state.generated_image = None

    if 'generated_adv' not in st.session_state:
        st.session_state.generated_adv = None

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
            return images
        else:
            print("Error: The images object is empty or does not contain a PIL Image object.")

    st.write("<h1 style='text-align: center; color: #D75850;'>🤖 Advertisement Creation 🤖</h1>",
             unsafe_allow_html=True)

    st1, st2 = st.columns(2, gap="medium")

    st1.write("<h3 style='text-align: center; color: #4F92D6;'>Image Generation</h3>", unsafe_allow_html=True)

    uploaded_file = st1.file_uploader("Select photo 📷", type=["jpg", "jpeg", "png"])

    prompt = st1.text_input("Enter the prompt: 💻")

    #color = st1.text_input("Enter the color (hex format): 🎨")
    color = st1.color_picker('Pick the color (hex format): 🎨', '#00f900')
    def generate_image(image, prompt, color):
        images = task1_imp(user_image=image, user_prompt=prompt, user_color=color)
        return images.images[0]

    if uploaded_file is not None:
        # Seçilen dosyayı aç
        image = Image.open(uploaded_file)

    if uploaded_file is not None and prompt != "" and color != "":
        button_clicked = st1.button("Generate Image", type="primary", use_container_width=True)
    else:
        st1.warning("Please upload a file, enter a prompt, and specify a color before generating.")
        button_clicked = st1.button("Generate Image", disabled=True, use_container_width=True)

    if button_clicked:
        st1.write("Generating image...")
        generated_image = generate_image(image=image, prompt=prompt, color=color)
        st.session_state.generated_image = generated_image

    if st.session_state.generated_image is not None:
        st1.image(st.session_state.generated_image, caption="Generated Image by @erendagstan", use_column_width=True, width=512)
        image_bytes = BytesIO()
        st.session_state.generated_image.save(image_bytes, format="PNG")
        st1.download_button("Download Image", image_bytes.getvalue(), key="download_button",
                            file_name="generated_image.png", mime="image/png")

    ########
    st2.write("<h3 style='text-align: center; color: #4F92D6;'>Advertisement Generation using Image</h3>",
              unsafe_allow_html=True)

    def cerceve(images):
        """
        Adds a white frame around the generated image
        :param images: PIL.Image -> generated image
        :return: PIL.Image -> with added frame
        """
        cerceve_genislik = 350
        # Çerçeve rengini belirle (255 beyaz için)
        cerceve_rengi = (255, 255, 255)
        # Yeni bir görsel oluştur
        yeni_genislik = images.width + 2 * cerceve_genislik
        yeni_yukseklik = images.height + 2 * cerceve_genislik + 100
        yeni_gorsel = Image.new("RGB", (yeni_genislik, yeni_yukseklik), cerceve_rengi)
        # Ana görseli yeni görselin içine kopyala
        yeni_gorsel.paste(images, (cerceve_genislik, cerceve_genislik))
        return yeni_gorsel

    def logo_add(yeni_gorsel, logo_path="C:/Users/ASUS/PycharmProjects/spaceship-titanic/logos/cland.png"):
        """
        Adds the specified logo on top of the generated image.
        :param yeni_gorsel: Image -> Image to add the logo to
        :param logo_path: String -> Path of the logo file you want to add.
        :return: PIL.Image.Image -> Image with added logo.
        """
        logo = Image.open(logo_path)
        # Fotoğrafın boyutlarını al
        foto_genislik, foto_yukseklik = yeni_gorsel.size
        # Logo boyutunu belirle
        logo_boyut = (230, 192)  # İhtiyaca göre değiştirilebilir
        kucuk_logo = logo.resize(logo_boyut)
        # Logo'yu ortalanmış konuma yerleştir
        x_konum = (foto_genislik - logo_boyut[0]) // 2
        y_konum = 50  # (foto_yukseklik - logo_boyut[1]) // 4  # Yukarı ortaya daha yakın bir konum
        yeni_gorsel.paste(kucuk_logo, (x_konum, y_konum))
        return yeni_gorsel

    def punchline_add(yeni_gorsel, yazi_metni="AI ad banners lead to higher\nconversions ratesxxxx",
                      yazi_rengi=(0, 0, 0)):
        """
        Adds the punchline text below the generated image.
        :param yeni_gorsel: (PIL.Image.Image) -> Image to add the text to.
        :param yazi_metni: string -> The text you want to add.
        :param yazi_rengi: string -> hex code & Text color, default is black (0, 0, 0).
        :return: PIL.Image.Image -> Image with added text.
        """
        # Yazı tipini ve boyutunu belirle
        yazi_tipi = ImageFont.truetype(
            "C:/Users/ASUS/PycharmProjects/spaceship-titanic/fonts/PlayfairDisplay-VariableFont_wght.ttf", size=70)
        print(yazi_metni)
        x_konum = (yeni_gorsel.width - yazi_tipi.size) // 2
        y_konum = yeni_gorsel.height - 400  # Fotoğrafın alt kısmına 50 piksel mesafe
        # Bir çizim nesnesi oluştur
        draw = ImageDraw.Draw(yeni_gorsel)
        for row in yazi_metni.split("\n"):
            yazi_genislik = draw.textlength(text=row, font=yazi_tipi)  # text=yazi_metni
            # Yazıyı çiz
            bb_l, bb_t, bb_r, bb_b = draw.textbbox((x_konum, y_konum), row)
            x = x_konum + (bb_r - bb_l) / 2
            y = y_konum + (bb_b - bb_t) / 2
            draw.text(((yeni_gorsel.width - yazi_genislik) // 2, int(y)), row, font=yazi_tipi, fill=yazi_rengi)
            yazi_yukseklik = 75
            y_konum += yazi_yukseklik
        return yeni_gorsel

    def add_button(yeni_gorsel, button_text, button_color="#316346", padding=10, corner_radius=20):
        """
        Adds a button the generated image
        :param yeni_gorsel: PIL.Image.Image
        :param button_text: string -> user button text
        :param button_color: string -> hex code
        :param padding: int -> default is 10
        :param corner_radius: int -> default is 20
        :return: return: PIL.Image.Image -> Image with added button.
        """
        # Görselin genişliği ve yüksekliğini al
        width, height = yeni_gorsel.size
        # Button boyutları ve konumu belirle
        button_width = 400
        button_height = 100
        button_x = (width - button_width) // 2
        button_y = height - 150
        # Bir çizim nesnesi oluştur
        draw = ImageDraw.Draw(yeni_gorsel)

        # Adjust the button dimensions for padding
        button_x += padding
        button_y += padding
        button_width -= 2 * padding
        button_height -= 2 * padding

        # Draw rounded rectangle for the button
        draw.rounded_rectangle(
            [button_x, button_y, button_x + button_width, button_y + button_height],
            fill=button_color,
            radius=corner_radius
        )

        # Buton üzerine metni çiz
        button_font_size = 30
        button_font = ImageFont.truetype("arial.ttf", size=button_font_size)

        # Metni ortalamak için textbbox fonksiyonunu kullanma
        bb_l, bb_t, bb_r, bb_b = draw.textbbox((button_x, button_y, button_x + button_width, button_y + button_height),
                                               button_text, font=button_font)

        # Yazının boyutlarını kontrol et
        text_width = bb_r - bb_l
        text_height = bb_b - bb_t

        # Yazıyı buton sınırları içinde sığdır
        max_text_width = button_width - 2 * padding
        max_text_height = button_height - 2 * padding

        if text_width > max_text_width:
            # Yazının genişliği buton genişliğini aşıyorsa, boyutunu küçült
            button_font_size = int(button_font_size * max_text_width / text_width)
            button_font = ImageFont.truetype("arial.ttf", size=button_font_size)

        if text_height > max_text_height:
            # Yazının yüksekliği buton yüksekliğini aşıyorsa, boyutunu küçült
            button_font_size = int(button_font_size * max_text_height / text_height)
            button_font = ImageFont.truetype("arial.ttf", size=button_font_size)

        # Metni ortalamak için textbbox fonksiyonunu kullanma
        bb_l, bb_t, bb_r, bb_b = draw.textbbox((button_x, button_y, button_x + button_width, button_y + button_height),
                                               button_text, font=button_font)

        text_x = button_x + (button_width - text_width) // 2
        text_y = button_y + (button_height - text_height) // 2
        draw.text((text_x, text_y), button_text, font=button_font, fill="white")

        return yeni_gorsel

    def task2_imp(image, logo_path, button_color, punchline_text, button_text):
        """
        Processes the images for Task 2 by adding a frame, logo, punchline, and a button.
        :param image: (PIL.Image.Image) ->  The input image(s) for processing.
        :param logo_path: string -> The logo path that user's selected
        :param button_color: string -> hex code
        :param punchline_text: string -> The text for the punchline.
        :param button_text: string -> The text for the button
        :return:
        """
        if image is not None:
            yeni_gorsel = cerceve(image)
            # logo
            yeni_gorsel = logo_add(yeni_gorsel, logo_path)
            # punchline
            yeni_gorsel = punchline_add(yeni_gorsel=yeni_gorsel, yazi_metni=punchline_text,
                                        yazi_rengi=button_color)  # yazi_metni="AI ad banners lead to higher\nconversions ratesxxxx"
            # button
            yeni_gorsel = add_button(yeni_gorsel, button_text=button_text, button_color=button_color)

            return yeni_gorsel
        else:
            print("Error: The images object is empty or does not contain a PIL Image object.")

    image_choosen = None
    st2_1, st2_2 = st2.columns(2, gap="small")
    image_chooser = st2_1.radio(
        "Which photo do you want to use? 👇",
        [":rainbow[Generated image]", "Add manual image 📁"],
        key="Generated image")

    if image_chooser == "Add manual image 📁":
        uploaded_file2 = st2_2.file_uploader("Select photo manually", type=["jpg", "jpeg", "png"])
        if uploaded_file2 is not None:
            user_manually_photo = Image.open(uploaded_file2)
            user_manually_photo = user_manually_photo.resize((512, 512))
            image_choosen = user_manually_photo
    else:
        if st.session_state.generated_image is not None:
            image_choosen = st.session_state.generated_image
            st2_2.write("<u>Generated image</u>", unsafe_allow_html=True)
            st2_2.image(image_choosen, width=120)

        # uploaded_file2 = st2_2.file_uploader("Select photo manually", type=["jpg", "jpeg", "png"])
        # image_choosen =

    logo_input = st2.text_input("Enter the logo path: 📂")
    #button_color_input = st2.text_input("Enter a color (hex format) for Button and Punchline: 🎨")
    button_color_input = st2.color_picker('Pick a color (hex format) for Button and Punchline: 🎨', '#00f900')
    punchline_text_input = st2.text_input(
        "Enter text for the punchline (You can use '\\n' to move to the bottom line): ✍")
    punchline_text_input = punchline_text_input.replace("\\n", "\n")
    button_text_input = st2.text_input("Enter text for the button: ✍")

    with st2:
        if image_choosen is not None and logo_input != "" and button_color_input != "" and punchline_text_input != "" and button_text_input != "":
            button_clicked2 = st.button("Generate Advertisement", type="primary", use_container_width=True)
        else:
            st.warning("Please upload an image or generate, enter a logo path, specify colors, and provide text before generating.")
            button_clicked2 = st.button("Generate Advertisement", disabled=True, use_container_width=True)

    if button_clicked2:
        advert_img = task2_imp(image=image_choosen, logo_path=logo_input,
                               button_color=button_color_input, punchline_text=punchline_text_input,
                               button_text=button_text_input)
        st.session_state.generated_adv = advert_img

    if st.session_state.generated_adv is not None:
        st2.image(st.session_state.generated_adv, caption="Generated Advertisement by @erendagstan", use_column_width=True)
        image_bytes2 = BytesIO()
        st.session_state.generated_adv.save(image_bytes2, format="PNG")
        st2.download_button("Download Advertisement", image_bytes2.getvalue(), key="download_button2",
                            file_name="generated_advertisement.png", mime="advertisement/png")


page_names_to_funcs = {
    "Homepage": intro,
    "Generate Advertisement": generation_page,
}

page_name = st.sidebar.selectbox("Choose a page", page_names_to_funcs.keys())
page_names_to_funcs[page_name]()
