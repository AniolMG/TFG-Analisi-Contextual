from bs4 import BeautifulSoup
import re
from time import sleep
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException
from fake_useragent import UserAgent
import urllib
from urllib.parse import urlparse, urljoin
import robotexclusionrulesparser
import random

from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException
from time import sleep
import os
import pickle
import undetected_chromedriver as uc
from bs4 import BeautifulSoup
from urllib.error import URLError
from multiprocessing import Process, Manager

def remove_curly_braces(text):
    # Define a regular expression pattern to find text within curly braces
    pattern = r'\{.*\}'
    
    # Use re.sub() to replace all occurrences of the pattern with an empty string
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text

def is_page_fully_loaded(driver):
    # Execute JavaScript to check if the document is ready
    return driver.execute_script("return document.readyState") == "complete"

def wait_for_consistent_load(driver, num_checks=6, interval=0.1):
    # This function makes sure that there are no asynchronous or pending loads that still haven't begun
    load_counts = 0
    for _ in range(num_checks):
        if is_page_fully_loaded(driver):
            load_counts += 1
            
        time.sleep(interval)
    
    return load_counts == num_checks

def save_cookies(driver, location):
    # Save cookies to a dictionary, keyed by domain
    cookies_dict = {}
    for cookie in driver.get_cookies():
        domain = cookie['domain']
        if domain not in cookies_dict:
            cookies_dict[domain] = []
        cookies_dict[domain].append(cookie)
    # Write the cookies dictionary to a file
    pickle.dump(cookies_dict, open(location, "wb"))

def load_cookies(driver, location, url):
    # Load the cookies dictionary from a file
    cookies_dict = pickle.load(open(location, "rb"))
    # Get the domain from the URL
    domain = urllib.parse.urlparse(url).netloc
    # If there are cookies for this domain, add them to the driver
    if domain in cookies_dict:
        for cookie in cookies_dict[domain]:
            driver.add_cookie(cookie)
            
def end_scroll(driver):
    body = driver.find_element("tag name", "body")
    time.sleep(random.uniform(1.04, 1.703))
    
    # Simulate pressing and holding the END key
    driver.execute_script("""
        var event = new KeyboardEvent('keydown', {
            key: 'End',
            code: 'End',
            charCode: 35,
            keyCode: 35,
            which: 35
        });
        document.dispatchEvent(event);
    """)
    time.sleep(random.uniform(0.07, 0.15))  # Hold the key down for X seconds

    # Simulate releasing the END key
    driver.execute_script("""
        var event = new KeyboardEvent('keyup', {
            key: 'End',
            code: 'End',
            charCode: 35,
            keyCode: 35,
            which: 35
        });
        document.dispatchEvent(event);
    """)
    
    time.sleep(random.uniform(0.73, 1.28))

# Function to fetch a URL and store the result in a shared dictionary
def fetch_url(rp, url, return_dict):
    try:
        # Attempt to fetch the URL using the rp.fetch (rp is a RobotExclusionRulesParser instance)
        rp.fetch(url)
        # If successful, mark success in the specified dictionary
        return_dict['success'] = True
    except Exception as e:
        # If there's an error, store the error message in the specified dictionary
        return_dict['error'] = str(e)

# Function to fetch a URL with a timeout mechanism
# Necessary because RobotExclusionRulesParser class doesn't have built-in handling for some errors
def rpFetchWithTimeout(rp, url, timeout=10):
    # Use a Manager to create a shared dictionary for communication between processes
    with Manager() as manager:
        return_dict = manager.dict() # Shared dictionary to store results or errors
        # Create a new process to fetch the URL
        p = Process(target=fetch_url, args=(rp, url, return_dict))
        p.start() # Start the process
        p.join(timeout) # Wait for the process to finish, with a timeout

        # If the process is still alive after the timeout
        if p.is_alive():
            print("Timeout en l'operació de fetch")
            p.terminate() # Terminate the process
            p.join() # Ensure the process has terminated
            raise URLError("Fetch operation timed out") # Raise a timeout error
        # If there was an error during fetching, handle it
        if 'error' in return_dict:
            #print(f"No s'ha pogut accedir a {url}: {return_dict['error']}")
            raise URLError(f"Failed to fetch {url}: {return_dict['error']}")

# Function to check if webscraping is allowed for a certain URL
def can_scrape(url, user_agent):
    # Parse the URL to extract the base components (scheme and netloc)
    parsed_url = urlparse(url)
    # Construct the "base" or "main" URL (example: "https://example.com")
    base_url = parsed_url.scheme + "://" + parsed_url.netloc
    # Construct the robots.txt URL (example: "https://example.com/robots.txt")
    robots_url = urljoin(base_url, "/robots.txt")

    # Initialize the robot exclusion rules parser
    rp = robotexclusionrulesparser.RobotExclusionRulesParser()
    # Set the user-agent for the parser
    rp.user_agent = user_agent
    print("Analitzant", robots_url)
    try:
        # Fetch the robots.txt file with a timeout
        rpFetchWithTimeout(rp, robots_url)
    except URLError as e:
        # If fetching fails, show an error and return False
        print(f"No s'ha pogut accedir a {robots_url}")
        # print(f"No s'ha pogut accedir a {robots_url}: {e}") # Full Message
        return False

    # Check if the URL allows webscraping
    return rp.is_allowed(user_agent, url)

def extract_text_from_url(url, max_retries=5, retry_delay=5):
    retries = 0
    cookies_location = "cookies.pkl"
    while retries < max_retries:
        try:
            # Configure Chrome options
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run in headless mode (without opening browser window)
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--crash-dumps-dir=/tmp") 
            chrome_options.add_argument("--shm-size=2g")
            
            # Initialize the UserAgent object
            ua = UserAgent()
            # Get a random User-Agent
            user_agent = ua.random
            print("User Agent:", user_agent)
            chrome_options.add_argument(f'user-agent={user_agent}')
            
            # Create the WebDriver and set the browser options
            driver = uc.Chrome(executable_path='./chromedriver123',headless=True,use_subprocess=False, options=chrome_options) # This version of ChromeDriver only supports Chrome version 122

            if(can_scrape(url, user_agent)): print("La pàgina permet fer webscraping.")
            else: 
                print("La pàgina no permet fer webscraping...")
                return

            time.sleep(random.uniform(0.51, 1.01))
            
            # Set window size to allow size dependent elements to be loaded
            # Also set large height to force all the webpage to load
            driver.set_window_size(1920, 17500)
            ##### Techniques to avoid bot detection
            # Report a more common resolution
            driver.execute_script("Object.defineProperty(screen, 'width', {get: function () { return 1920; }});")
            driver.execute_script("Object.defineProperty(screen, 'height', {get: function () { return 1080; }});")
            # Change the language
            driver.execute_script("Object.defineProperty(navigator, 'language', {get: function () { return 'en-US'; },});")
            
            # Change the platform
            driver.execute_script("Object.defineProperty(navigator, 'platform', {get: function () { return 'Win32'; },});")
            
            #####
            # Navigate to the URL
            print("Navegant a l'enllaç...")
            # Changing the property of the navigator value for webdriver to undefined 
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            # Set the page load timeout
            driver.set_page_load_timeout(15) # Considered a timeout if it takes more than 15s
            time.sleep(0.15)
            driver.get(url)
            print("HTML inicial carregat!")
            # Load cookies
            if os.path.exists(cookies_location):
                load_cookies(driver, cookies_location, url)
            
            # Scroll to the bottom of the page
            time.sleep(random.uniform(1.5, 2.2))
            end_scroll(driver)
            # Wait until the page is fully loaded
            WebDriverWait(driver, 10).until(wait_for_consistent_load)
            print("Contingut dinàmic carregat!")
            time.sleep(random.uniform(0.9,1.4)) # Random time to try to avoid bot detection

            save_cookies(driver, cookies_location)
            
            # Find the body element
            body = driver.find_element("tag name", "body")
            time.sleep(random.uniform(0.5,1))
            
            # Execute JavaScript to hide all fixed and sticky elements, and dialogs or popups, that contain certain keywords (among other elements)
            driver.execute_script("""

                function traverseNodes(node, elements) {
                    var allElements = node.querySelectorAll('*');
                    for (var i = 0; i < allElements.length; i++) {
                        elements.push(allElements[i]);
                    }
                
                    // Check for shadow roots and traverse into them
                    var shadowRoots = node.querySelectorAll('*');
                    shadowRoots.forEach(function(shadowRoot) {
                        var shadowTree = shadowRoot.shadowRoot;
                        if (shadowTree) {
                            traverseNodes(shadowTree, elements);
                        }
                    });
                }


                            
                var elements = []; //document.body.getElementsByTagName('*');
                traverseNodes(document.body, elements); 
                var items = []; 
                var words = ['policy', 'cooki', 'políti', 'galet', 'gallet', 'accept', 'consent', 'uc-'];
                var ariaAttributes = ['aria-describedby', 'sr-only'];
                for (var i = 0; i < elements.length; i++) {
                    var element = elements[i];
                    var attributes = element.attributes;
                    var computedStyle = window.getComputedStyle(element);
                    
                    var hasAriaAttribute = ariaAttributes.some(attr => element.hasAttribute(attr) || element.className && element.className.toString().includes(attr)) ;
                    var attributeContainsDialogOrPopup = attributes && Array.from(attributes).some(attr => attr.value.includes('dialog') || attr.value.includes('popup'));
                    
                    //"Invisible" Text
                    if(computedStyle.getPropertyValue('height') === '1px' || computedStyle.getPropertyValue('width') === '1px'){
                        items.push(element);
                    }
                    //Cookie Banners
                    else if ((getComputedStyle(element).position === 'fixed' || getComputedStyle(element).position === 'sticky'
                    || element.getAttribute('role') === "dialog"/*|| attributeContainsDialogOrPopup*/) && 
                        words.some(word => element.innerText && element.innerText.includes(word))) {
                        items.push(element);
                    }
                    else if (words.some(word => (element.className && element.className.toString().includes(word) || element.tagName && element.tagName.includes(word)
                    )) &&
                    words.some(word => element.innerText && element.innerText.includes(word))) {
                        items.push(element);
                    }
                    else if(element.className && element.className.toString().includes('addoor')){
                        items.push(element);
                    }
                    else if (element.getAttribute('role') === "dialog") {
                        items.push(element);
                    }
                }

                for (var i = 0; i < items.length; i++) {
                    items[i].parentNode.removeChild(items[i]);
                }
            """)
            # print("Text cleaned!") # Debugging
            # Extract text content using Selenium
            text = driver.find_element(By.TAG_NAME, 'body').text
            # print("Fetched cleaned body!") # Debugging
            # Remove curly braces
            text = remove_curly_braces(text)
            #print("Removed curly braces!") # Debugging
            print("Text netejat!")
            
            # Check if the number of words is less than 250
            if len(text.split()) < 250:
                # If the fetched text is too short, we'll add the text from a less sophisticated approach
                # In some cases, this will be much more useful
                
                # Get the page source HTML
                html = driver.page_source

                # Parse the HTML with BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                
                # Find the body element and get its text
                soup_text = soup.body.get_text()
            
                # Append the new text to the existing text
                text += " " + soup_text
            
                # Now 'text' contains the text of the body element
                print("Cos tornat a obtenir, ara amb BeautifulSoup!")
            
            return text

        except (TimeoutException, WebDriverException) as e:
            print(f"Error occurred while fetching the URL: {e}")
            retries += 1
            if retries < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                sleep(retry_delay)

        finally:
            try:
                # Close the WebDriver
                driver.quit()
            except UnboundLocalError:
                pass

    print("Maximum number of retries reached. Unable to fetch the URL:", url)
    return None
