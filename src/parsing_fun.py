import itertools
import math
import os
import random
import re
import time
import warnings
from functools import partial

import pandas as pd
import pydub
import requests
import speech_recognition as sr
from bs4 import BeautifulSoup
from p_tqdm import p_map
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from seleniumwire import webdriver as wire_webdriver
from tqdm import tqdm, tqdm_notebook
from twocaptcha import TwoCaptcha

warnings.filterwarnings("ignore")

def proxy():
    """
    Function to create a Chrome browser instance with proxy server settings and
    disabling unnecessary features to improve performance.

    Description:
    -----------
    1. Necessary options are set for the Chrome browser:
       - `--disable-blink-features=AutomationControlled`: prevents detection 
       of automated browser control (hides the use of Selenium).
       - `--disable-javascript`: disables JavaScript execution to speed up 
       page loading and reduce resource usage (can be disabled for sites where JavaScript is required).

    2. An instance of `webdriver.Chrome` is created, which can be used for 
    automated browser control.
       - The `--headless` option can be uncommented to run the browser in 
       headless mode without a graphical interface.

    3. Proxy servers are connected using the `proxy_rotator` library. 
    The proxy is fetched via `get_proxy()` and set for the browser using the 
    `seleniumwire_options`.

    Returns:
    --------
    - A Chrome browser instance (`webdriver.Chrome`) with proxy settings and 
      JavaScript disabled.
    """
    
    # Browser options
    chrome_options = Options()

    # Launch in headless mode without a graphical interface (can be removed for visual control)
    # chrome_options.add_argument("--headless")

    # Hide automation
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")

    # Disable JavaScript
    chrome_options.add_argument("--disable-javascript")

    # Specify the path to ChromeDriver if it is not in PATH
    service = Service(executable_path="")

    # Create a Chrome browser instance
    driver = webdriver.Chrome(service=service, options=chrome_options)

    proxy = proxy_rotator.get_proxy()
    # print(f"using proxy: {proxy}")
    prx = {}
    prx.update({"proxy": proxy})
    print(prx)

    # Create a driver with proxy authorization
    driver = wire_webdriver.Chrome(seleniumwire_options=prx, options=chrome_options)
    return driver
   

class ProxyRotator:
    """
    Class for rotating proxy servers during requests.

    Attributes:
    ----------
    proxies : list
        List of proxy servers for rotation.
    current_proxy : str
        Currently used proxy server.
    good_proxy : dict
        Dictionary to store good proxy servers (optional).
    request_counter : int
        Counter to determine when to switch the proxy server.

    Methods:
    -------
    change_proxy():
        Switches the current proxy server to the next one in the cycle.
    get_proxy():
        Returns the current proxy server and switches it if needed.
    """

    def __init__(self, proxies):
        """
        Initializes ProxyRotator with the given list of proxy servers.

        Parameters:
        ----------
        proxies : list
            List of proxy servers for rotation.
        """
        self.proxies = itertools.cycle(proxies)
        self.current_proxy = None
        self.good_proxy = {}
        self.request_counter = 0

    def change_proxy(self):
        """
        Switches the currently used proxy server to the next one in the cycle.

        Returns:
        ----------
        str:
            The new current proxy server.
        """
        self.current_proxy = next(self.proxies)
        self.request_counter = 0
        return self.current_proxy

    def get_proxy(self):
        """
        Returns the current proxy server and checks if it needs to be switched.

        The proxy server will be changed after every second request.

        Returns:
        ----------
        str:
            The current or new proxy server.
        """
        if self.request_counter % 2 == 0:
            self.change_proxy()
        self.request_counter += 1
        return self.current_proxy


def random_sleep(min_time=1, max_time=3):
    """
    Function for introducing a random delay in program execution.

    Pauses execution for a random number of seconds between `min_time` and `max_time`.

    Parameters:
    ----------
    min_time : int or float, optional
        Minimum delay time in seconds (default is 1 second).
    max_time : int or float, optional
        Maximum delay time in seconds (default is 3 seconds).
    
    Returns:
    ----------
    None
    """
    time.sleep(random.uniform(min_time, max_time))


def captcha_bypass(browser):
    """
    Function for automatically bypassing CAPTCHA on a web page.

    Interacts with the CAPTCHA, including switching to an iframe, clicking CAPTCHA elements,
    and handling audio CAPTCHA. If an error occurs, the browser restarts with a new proxy.

    Parameters:
    ----------
    browser : selenium.webdriver.Chrome
        Browser instance where the CAPTCHA page is loaded.

    Returns:
    ----------
    browser : selenium.webdriver.Chrome
        Browser instance after CAPTCHA processing.

    Exceptions:
    -----------
    If an error occurs while processing the CAPTCHA, the driver restarts with a new proxy.
    """

    try:
        # Start CAPTCHA processing
        print("Starting CAPTCHA processing")

        # Set implicit wait time
        browser.implicitly_wait(20)

        # Find all iframes on the page
        frames = browser.find_elements(By.TAG_NAME, "iframe")

        # Switch to the CAPTCHA iframe
        WebDriverWait(browser, 20).until(
            EC.frame_to_be_available_and_switch_to_it(
                (By.CSS_SELECTOR, f"iframe[title='reCAPTCHA']")
            )
        )

        # Click the CAPTCHA checkbox
        WebDriverWait(browser, 20).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="recaptcha-anchor"]/div[1]'))
        ).click()

        browser.implicitly_wait(20)

        # Switch back to the main page context
        browser.switch_to.default_content()

        # Switch to the iframe with the image verification challenge
        WebDriverWait(browser, 20).until(
            EC.frame_to_be_available_and_switch_to_it(
                (
                    By.CSS_SELECTOR,
                    f"iframe[title='You can complete this CAPTCHA within two minutes']",
                )
            )
        )

        # Click the audio CAPTCHA button
        WebDriverWait(browser, 20).until(
            EC.element_to_be_clickable((By.XPATH, '//*[@id="recaptcha-audio-button"]'))
        ).click()

        # Retrieve the CAPTCHA audio source
        src = browser.find_element(By.ID, "audio-source").get_attribute("src")
        print(f"[INFO] Audio src: {src}")

        # Process the audio and input the resulting text
        write_audio(src)
        browser.find_element(By.CSS_SELECTOR, 'input[id="audio-response"]').send_keys(
            audio_to_text().lower()
        )
        browser.find_element(By.ID, "audio-response").send_keys(Keys.ENTER)

        # Wait for CAPTCHA verification to complete
        time.sleep(10)

    except Exception as ex:
        print(f"CAPTCHA encountered an error: {ex}")

        # Close the browser and restart with a new proxy
        browser.quit()
        time.sleep(2)
        browser = proxy()

    return browser


def write_audio(url):
    """
    Downloads an audio file from the specified URL and saves it to the local disk.

    Parameters:
    ----------
    url : str
        The URL of the audio file to be downloaded.

    Returns:
    ----------
    None
        This function does not return any values but saves the audio file in the current directory.

    Exceptions:
    -----------
    If the request to the URL fails, it raises a `requests.exceptions.RequestException`.
    """

    # Perform the request to retrieve the audio file
    response = requests.get(url)

    # Save the audio file to disk
    with open(f"{name_audio_file}", "wb") as file:
        file.write(response.content)


def audio_to_text():
    """
    Converts an audio file from MP3 to WAV format and extracts text from the audio.

    Process:
    --------
    1. Opens the MP3 file.
    2. Converts it to WAV format.
    3. Uses the `SpeechRecognition` library to convert speech to text.

    Returns:
    ----------
    key : str
        The recognized text extracted from the audio file.

    Exceptions:
    -----------
    If recognition fails, it may raise `sr.UnknownValueError` or `sr.RequestError`.

    Dependencies:
    ------------
    - pydub.AudioSegment: For audio format conversion.
    - speech_recognition.Recognizer: For converting audio to text.
    """

    # Open the MP3 file
    with open(f"{name_audio_file}", "rb") as files:
        path_to_mp3 = name_audio_file
        path_to_wav = "audio_file_wav.wav"

        # Convert MP3 to WAV
        sound = pydub.AudioSegment.from_file(path_to_mp3, "mp3")
        sound.export(path_to_wav, format="wav")

        # Prepare the audio file for recognition
        sample_audio = sr.AudioFile(path_to_wav)
        r = sr.Recognizer()

        # Read the audio and recognize the text
        with sample_audio as source:
            audio = r.record(source)

        # Convert audio to text
        key = r.recognize_google(audio)

        return key


def viewbull_field_container(page_source, label_name):
    """
    Extracts a value from an HTML container based on a label using BeautifulSoup.

    Parameters:
    ----------
    page_source : str
        HTML content of the page to be parsed.
    label_name : str
        Text contained in the desired label to identify the corresponding container.

    Returns:
    -----------
    value : str or None
        The text value extracted from the container with the class 'value' corresponding to the label 'label_name'.
        Returns None if the container is not found or the label does not match.

    Description:
    ---------
    1. Creates a BeautifulSoup object for HTML parsing.
    2. Finds all containers with the class 'field viewbull-field__container'.
    3. Iterates through each container, looking for a <div> with the class 'label' containing the 'label_name'.
    4. If a container with the desired label is found, extracts text from the <div> with the class 'value'.
    5. Returns the extracted value or None if nothing is found.

    Example usage:
    ---------------------
    value = viewbull_field_container(page_source, "Label Name")
    """
    
    # Create BeautifulSoup object for parsing
    soup = BeautifulSoup(page_source, "html.parser")

    # Find all containers with the class 'field viewbull-field__container'
    containers = soup.find_all("div", class_="field viewbull-field__container")

    # Search for the container where <div class="label"> contains the text label_name
    value = None
    for container in containers:
        label = container.find("div", class_="label")
        if label and label.text.strip() == label_name:  # Check text inside <div class="label">
            value = container.find("div", class_="value")  # Find corresponding <div class="value">
            if value:
                value = value.text.strip()  # Extract text from <div class="value">
            break

    return value


def process_page(soup, i, data_df_combined):
    """
    Parses the HTML content of a listing page and extracts information to populate a DataFrame.

    Parameters:
    ----------
    soup : BeautifulSoup
        BeautifulSoup object containing the HTML content of the page to parse.
    i : int
        Index of the current row in the DataFrame where data will be recorded.
    data_df_combined : pd.DataFrame
        DataFrame to which the extracted data will be added.

    Description:
    ---------
    The function extracts various pieces of information from the web page, such as:
    - Publication date of the listing.
    - Object price.
    - Information on the district, address, house type, and other property parameters.
    - Year of commissioning, construction status, mortgage availability.
    - Property description.

    All extracted data is stored in the passed DataFrame at the corresponding positions using the index i.

    Execution steps:
    ----------------
    1. Extracts the publication date and saves it to the "data" field.
    2. Extracts the object price and saves it to the "price" field.
    3. Uses the helper function `viewbull_field_container` to extract various text fields.
    4. Processes fields like district, address, house type, apartment type, window side, renovation, documented area, floor, balcony.
    5. Extracts the year of commissioning, if available.
    6. Extracts the construction status and mortgage availability.
    7. Extracts the property description (multiple variations).

    Example usage:
    ---------------------
    process_page(soup, i, data_df_combined)
    """
    
    # Parse the page
    # Publication date
    try:
        data_df_combined.at[i, "data"] = soup.find(
            "div", class_="viewbull-actual-date"
        ).text
    except:
        data_df_combined.at[i, "data"] = None

    # Price
    price_element = soup.find("span", class_="viewbull-summary-price__value")
    price = price_element["data-bulletin-price"] if price_element else None
    if price is not None:
        price = re.match(r"\d+", price)
        price = int(price.group()) if price else None
    data_df_combined.at[i, "price"] = price

    # Extract values using the helper function
    data_df_combined.at[i, "district_value"] = viewbull_field_container(page_source, "Район")
    data_df_combined.at[i, "address_value"] = viewbull_field_container(page_source, "Адрес")
    data_df_combined.at[i, "house_type_value"] = viewbull_field_container(page_source, "Тип дома")
    data_df_combined.at[i, "apartment_type"] = viewbull_field_container(page_source, "Вид квартиры")
    data_df_combined.at[i, "window_side"] = viewbull_field_container(page_source, "Сторона окон")
    data_df_combined.at[i, "renovation"] = viewbull_field_container(page_source, "Ремонт")
    data_df_combined.at[i, "area_documents"] = viewbull_field_container(page_source, "Площадь по документам")
    data_df_combined.at[i, "floor"] = viewbull_field_container(page_source, "Этаж")
    data_df_combined.at[i, "balcony_loggia"] = viewbull_field_container(page_source, "Балкон (лоджия)")

    # Year of commissioning
    label = soup.find("div", class_="label", string="Год ввода в эксплуатацию")
    if label:
        value_div = label.find_next("div", class_="value")
        if value_div:
            year = value_div.find("span").text.strip()
            data_df_combined.at[i, "year"] = year

    # Construction status
    try:
        data_df_combined.at[i, "construction_status"] = soup.find(
            "span", {"data-field": "constructionStatus"}
        ).text
    except:
        data_df_combined.at[i, "construction_status"] = None

    # Mortgage
    try:
        data_df_combined.at[i, "mortgage"] = soup.find(
            "span", {"data-field": "mortgage"}
        ).text
    except:
        data_df_combined.at[i, "mortgage"] = None

    # Description
    try:
        data_df_combined.at[i, "description"] = str(soup.find_all("p", class_="inplace"))
    except:
        data_df_combined.at[i, "description"] = None

    try:
        data_df_combined.at[i, "description2"] = soup.find("p", class_="inplace auto-shy").text
    except:
        data_df_combined.at[i, "description2"] = None
