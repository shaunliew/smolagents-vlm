from io import BytesIO
from time import sleep

import helium
from dotenv import load_dotenv
from PIL import Image
from selenium import webdriver
from selenium.common.exceptions import ElementNotInteractableException, TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.keys import Keys
from smolagents import CodeAgent, LiteLLMModel, OpenAIServerModel, TransformersModel, tool  # noqa: F401
from smolagents.agents import ActionStep
load_dotenv()
import os
import streamlit as st
import json


# Let's use Qwen-2VL-72B via an inference provider like Fireworks AI

model = OpenAIServerModel(
    api_key=os.getenv("FIREWORKS_API_KEY"),
    api_base="https://api.fireworks.ai/inference/v1",
    model_id="accounts/fireworks/models/qwen2-vl-72b-instruct",
)

# You can also use a close model

# model = LiteLLMModel(
#     model_id="gpt-4o",
#     api_key=os.getenv("OPENAI_API_KEY"),
# )

# locally a good candidate is Qwen2-VL-7B-Instruct
# model = TransformersModel(
#     model_id="Qwen/Qwen2-VL-7B-Instruct",
#     device_map = "auto",
#     flatten_messages_as_text=False
# )


# Prepare callback
def save_screenshot(step_log: ActionStep, agent: CodeAgent) -> None:
    sleep(1.0)  # Let JavaScript animations happen before taking the screenshot
    driver = helium.get_driver()
    current_step = step_log.step_number
    if driver is not None:
        for step_logs in agent.logs:  # Remove previous screenshots from logs for lean processing
            if isinstance(step_log, ActionStep) and step_log.step_number <= current_step - 2:
                step_logs.observations_images = None
        png_bytes = driver.get_screenshot_as_png()
        image = Image.open(BytesIO(png_bytes))
        print(f"Captured a browser screenshot: {image.size} pixels")
        step_log.observations_images = [image.copy()]  # Create a copy to ensure it persists, important!

    # Update observations with current URL
    url_info = f"Current url: {driver.current_url}"
    step_log.observations = url_info if step_logs.observations is None else step_log.observations + "\n" + url_info
    return


# Initialize driver only when needed
def initialize_driver():
    chrome_options = webdriver.ChromeOptions()
    
    # Make automation less detectable
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    
    # Add realistic user agent
    chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
    
    # Additional settings to reduce bot detection
    chrome_options.add_argument('--disable-infobars')
    chrome_options.add_argument('--start-maximized')
    chrome_options.add_argument('--disable-popup-blocking')
    chrome_options.add_argument("--force-device-scale-factor=1")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-pdf-viewer")
    
    # Create CDP capabilities to modify navigator.webdriver flag
    chrome_options.add_argument('--remote-debugging-port=9222')
    
    driver = helium.start_chrome(headless=False, options=chrome_options)
    return driver

# Initialize tools
@tool
def search_item_ctrl_f(text: str, nth_result: int = 1) -> str:
    """
    Searches for text on the current page via Ctrl + F and jumps to the nth occurrence.
    Args:
        text: The text to search for
        nth_result: Which occurrence to jump to (default: 1)
    """
    elements = driver.find_elements(By.XPATH, f"//*[contains(text(), '{text}')]")
    if nth_result > len(elements):
        raise Exception(f"Match n°{nth_result} not found (only {len(elements)} matches found)")
    result = f"Found {len(elements)} matches for '{text}'."
    elem = elements[nth_result - 1]
    driver.execute_script("arguments[0].scrollIntoView(true);", elem)
    result += f"Focused on element {nth_result} of {len(elements)}"
    return result


@tool
def go_back() -> None:
    """Goes back to previous page."""
    driver.back()


@tool
def close_popups() -> str:
    """
    Closes any visible modal or pop-up on the page. Use this to dismiss pop-up windows! This does not work on cookie consent banners.
    """
    # Common selectors for modal close buttons and overlay elements
    modal_selectors = [
        "button[class*='close']",
        "[class*='modal']",
        "[class*='modal'] button",
        "[class*='CloseButton']",
        "[aria-label*='close']",
        ".modal-close",
        ".close-modal",
        ".modal .close",
        ".modal-backdrop",
        ".modal-overlay",
        "[class*='overlay']",
    ]

    wait = WebDriverWait(driver, timeout=0.5)

    for selector in modal_selectors:
        try:
            elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, selector)))

            for element in elements:
                if element.is_displayed():
                    try:
                        # Try clicking with JavaScript as it's more reliable
                        driver.execute_script("arguments[0].click();", element)
                    except ElementNotInteractableException:
                        # If JavaScript click fails, try regular click
                        element.click()

        except TimeoutException:
            continue
        except Exception as e:
            print(f"Error handling selector {selector}: {str(e)}")
            continue
    return "Modals closed"

@tool
def input_search(text: str, submit: bool = True) -> str:
    """
    Inputs text into a search box and optionally submits the search.
    Args:
        text: The text to input into the search box
        submit: Whether to submit the search (default: True)
    Returns:
        str: Status message indicating success or failure
    """
    try:
        # Common search box selectors, ordered from most specific to most general
        search_selectors = [
            # FairPrice specific selectors
            "#search-input-bar",
            "[data-testid='search-input-desktop']",
            
            # Lazada specific selectors
            ".search-box__input--O34g",
            ".search-box__input",
            
            # Generic selectors
            "[type='search']",
            "[name='search']",
            "[name='q']",
            "[name='query']",
            "[placeholder*='search' i]",
            "[placeholder*='Search' i]",
            "[aria-label*='search' i]",
            ".search-input",
            "#search",
            ".searchbox",
            "[role='search'] input"
        ]
        
        wait = WebDriverWait(driver, timeout=3)
        
        # Try each selector until we find a visible search box
        for selector in search_selectors:
            try:
                search_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))
                if search_box.is_displayed():
                    # Ensure the element is interactable
                    wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                    
                    # Clear existing text
                    search_box.clear()
                    
                    # Input new text
                    search_box.send_keys(text)
                    sleep(0.5)  # Small delay to ensure text is entered
                    
                    if submit:
                        # Try pressing Enter to submit
                        search_box.send_keys(Keys.RETURN)
                        sleep(1)  # Wait for search to initiate
                    
                    return f"Successfully input '{text}' into search box using selector: {selector}"
            except TimeoutException:
                continue
            except ElementNotInteractableException:
                print(f"Element not interactable with selector {selector}, trying next...")
                continue
            except Exception as e:
                print(f"Error with selector {selector}: {str(e)}")
                continue
                
        raise Exception("No search box found after trying all selectors")
        
    except Exception as e:
        return f"Failed to input search text: {str(e)}"

@tool
def final_answer(answer: str) -> str:
    """
    Returns the final answer for the task.
    Args:
        answer: The final answer to return
    Returns:
        str: The provided answer
    """
    return f"Out - Final answer: {answer}"

@tool
def click_product_image(product_name: str = "iPhone 15 Pro Max") -> str:
    """
    Clicks on a product image or link based on product name across different e-commerce websites.
    Args:
        product_name: The name of the product to look for
    Returns:
        str: Status message indicating success or failure
    """
    try:
        # Detect which site we're on
        current_url = driver.current_url.lower()
        is_lazada = 'lazada' in current_url
        is_fairprice = 'fairprice' in current_url
        
        # Split product name into keywords for more flexible matching
        keywords = product_name.lower().split()
        
        # Create a complex XPath that looks for elements containing all keywords
        keyword_conditions = []
        for keyword in keywords:
            keyword_conditions.append(f"contains(translate(., 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz'), '{keyword}')")
        
        keyword_xpath = " and ".join(keyword_conditions)
        
        # Wait for page load
        sleep(2)
        
        if is_lazada:
            # Lazada-specific selectors in order of preference
            selectors = [
                # LazMall section (first product)
                "(//div[contains(@class, 'Bm3ON') or contains(@class, 'grid-card')])[1]",
                "(//a[contains(@href, '//www.lazada.sg/products/')])[1]",
                "(//div[contains(@data-tracking-exposed-item-id, '')])[1]",
                "(//img[@type='product'])[1]/.."
            ]
        elif is_fairprice:
            # FairPrice-specific selectors
            selectors = [
                # Product card with exact match
                f"(//div[@data-testid='product'][.//span[contains(text(), '{product_name}')]])[1]",
                # Product card with name match
                f"(//div[@data-testid='product-card'][.//span[{keyword_xpath}]])[1]",
                # Product name link
                f"(//a[.//span[{keyword_xpath}]][@href])[1]",
                # Generic product card
                "(//div[@data-testid='product'])[1]",
                # Any product card with matching text
                f"(//div[contains(@class, 'product')]//span[{keyword_xpath}]/ancestor::div[contains(@class, 'product-card')])[1]"
            ]
        else:
            # Generic selectors as fallback
            selectors = [
                f"//a[{keyword_xpath}]",
                f"//div[{keyword_xpath}]//a",
                f"//img[{keyword_xpath}]/.."
            ]
        
        wait = WebDriverWait(driver, timeout=5)
        
        # Try each selector
        for selector in selectors:
            try:
                # Wait for elements to be present
                elements = driver.find_elements(By.XPATH, selector)
                
                if elements:
                    element = elements[0]  # Always take the first element
                    if element.is_displayed():
                        try:
                            # Scroll element into view with offset
                            driver.execute_script("""
                                arguments[0].scrollIntoView(true);
                                window.scrollBy(0, -100);
                            """, element)
                            sleep(0.5)
                            
                            # For FairPrice, try to find the clickable link first
                            if is_fairprice:
                                try:
                                    # Look for link within the product card
                                    links = element.find_elements(By.XPATH, ".//a[@href]")
                                    if links:
                                        links[0].click()
                                        return "Successfully clicked FairPrice product link"
                                except:
                                    pass
                            
                            # Try direct click if it's an anchor
                            if element.tag_name == 'a':
                                element.click()
                                return "Successfully clicked product link"
                            
                            # Try to find and click parent anchor
                            parent = element
                            max_iterations = 5
                            iterations = 0
                            
                            while parent and parent.tag_name != 'body' and iterations < max_iterations:
                                if parent.tag_name == 'a':
                                    parent.click()
                                    return "Successfully clicked product link"
                                try:
                                    parent = parent.find_element(By.XPATH, '..')
                                except:
                                    break
                                iterations += 1
                            
                            # If no anchor found, try direct click
                            element.click()
                            return "Successfully clicked product element"
                            
                        except Exception as click_error:
                            print(f"Click attempt failed: {str(click_error)}")
                            try:
                                # Try JavaScript click as last resort
                                driver.execute_script("arguments[0].click();", element)
                                return "Successfully clicked product with JavaScript"
                            except:
                                continue
            
            except Exception as e:
                print(f"Error with selector {selector}: {str(e)}")
                continue
                
        raise Exception(f"Could not find clickable element for {product_name}")
        
    except Exception as e:
        return f"Failed to click product: {str(e)}"

@tool
def get_product_details() -> str:
    """
    Extracts product name, price and promotion information from the current product page.
    Returns:
        str: JSON-formatted product details
    """
    try:
        sleep(2)  # Wait for fresh content to load
        
        # Initialize the product details dictionary with default values
        product_details = {
            "product": "Product name not found",
            "originalPrice": None,
            "currentPrice": "Price not found",
            "promotion": None
        }

        def extract_price(price_str: str) -> float:
            if not price_str or '$' not in price_str:
                return 0.0
            try:
                # Remove $ and commas, then convert to float
                price = float(price_str.replace('$', '').replace(',', ''))
                return price if price > 0 else 0.0
            except ValueError:
                return 0.0

        def format_price(price: float) -> str:
            return f"${price:.2f}"

        def validate_prices(current: float, original: float) -> tuple[float, float]:
            """
            Validates price logic and handles cache issues
            Returns tuple of (original_price, current_price)
            """
            if original <= 0:  # If original price not found
                return None, current
            if original < current:  # Likely cached/invalid original price
                return current, current
            return original, current

        # Selectors dictionary
        selectors = {
            'name': [
                # FairPrice specific selectors
                "span.sc-aa673588-1[weight='regular'][color='#333333']",
                ".sc-aa673588-1.drdope",
                "[data-testid='product-name-and-metadata'] span[weight='regular']",
                # Lazada specific selectors
                ".pdp-mod-product-badge-title",
                "h1.pdp-mod-product-title",
                # Generic selectors
                "h1",
                "[class*='product-name']",
                "[class*='title']:not([class*='promo'])"
            ],
            'current_price': [
                # FairPrice specific selectors
                "span.kQDEta.gbCpHo",
                "span.sc-aa673588-1.sc-6ac8ef58-5",
                # Lazada selectors
                ".pdp-price_type_normal",
                ".pdp-price",
                # Generic selectors
                "[class*='price']:not([class*='original']):not([class*='was'])"
            ],
            'original_price': [
                # FairPrice specific selectors
                "span.kZssPC",
                "span.sc-aa673588-1.kZssPC",
                # Lazada selectors
                ".pdp-price_type_deleted",
                ".pdp-price__old",
                # Generic selectors
                "[class*='original']",
                "[class*='was-price']"
            ]
        }

        # Find product name with validation
        for selector in selectors['name']:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if element.is_displayed():
                        text = element.text.strip()
                        if (len(text) > 5 and 
                            "Add to cart" not in text.lower() and 
                            "price" not in text.lower() and
                            "$" not in text):
                            product_details["product"] = text
                            break
                if product_details["product"] != "Product name not found":
                    break
            except Exception as e:
                print(f"Error with name selector {selector}: {str(e)}")
                continue

        # Initialize price variables
        current_price_value = 0.0
        original_price_value = 0.0

        # Find current price first
        for selector in selectors['current_price']:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if element.is_displayed():
                        price_text = element.text.strip()
                        if '$' in price_text:
                            current_price_value = extract_price(price_text)
                            if current_price_value > 0:
                                product_details["currentPrice"] = format_price(current_price_value)
                                break
                if current_price_value > 0:
                    break
            except:
                continue

        # Try to find original price
        for selector in selectors['original_price']:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                for element in elements:
                    if element.is_displayed():
                        price_text = element.text.strip()
                        if '$' in price_text:
                            original_price_value = extract_price(price_text)
                            if original_price_value > 0:
                                break
                if original_price_value > 0:
                    break
            except:
                continue

        # Validate and set prices
        original_price_value, current_price_value = validate_prices(current_price_value, original_price_value)
        
        # Update product details with validated prices
        if original_price_value:
            product_details["originalPrice"] = format_price(original_price_value)
            
            # Calculate promotion only if original price is higher than current price
            if original_price_value > current_price_value:
                savings = original_price_value - current_price_value
                product_details["promotion"] = format_price(savings)
        else:
            product_details["originalPrice"] = None
            product_details["promotion"] = None

        # Convert dictionary to formatted JSON string
        import json
        return json.dumps(product_details, indent=2)
        
    except Exception as e:
        return json.dumps({
            "product": "Product name not found",
            "originalPrice": None,
            "currentPrice": "Price not found",
            "promotion": None,
            "error": f"Failed to extract product details: {str(e)}"
        }, indent=2)
@tool
def handle_recaptcha() -> str:
    """
    Handles reCAPTCHA by finding and clicking the checkbox if it appears.
    Returns:
        str: Status message indicating success or failure
    """
    try:
        # First try to find the reCAPTCHA iframe
        wait = WebDriverWait(driver, timeout=3)
        recaptcha_frames = driver.find_elements(By.CSS_SELECTOR, "iframe[title*='reCAPTCHA']")
        
        if not recaptcha_frames:
            return "No reCAPTCHA found"
            
        # Try each frame that might contain the reCAPTCHA
        for frame in recaptcha_frames:
            try:
                # Switch to the frame
                driver.switch_to.frame(frame)
                
                # Look for the checkbox using various selectors
                checkbox_selectors = [
                    ".recaptcha-checkbox-border",
                    "#recaptcha-anchor",
                    "[role='checkbox']",
                    ".recaptcha-checkbox"
                ]
                
                for selector in checkbox_selectors:
                    try:
                        # Wait for checkbox to be clickable
                        checkbox = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
                        
                        if checkbox and checkbox.is_displayed():
                            # Try different click methods
                            try:
                                # Regular click
                                checkbox.click()
                            except:
                                try:
                                    # JavaScript click
                                    driver.execute_script("arguments[0].click();", checkbox)
                                except:
                                    continue
                                    
                            # Wait briefly to see if it worked
                            sleep(1)
                            
                            # Check if checkbox is now checked
                            checkbox_state = driver.execute_script(
                                "return arguments[0].getAttribute('aria-checked')", checkbox)
                            
                            if checkbox_state == 'true':
                                driver.switch_to.default_content()
                                return "Successfully clicked reCAPTCHA checkbox"
                    except:
                        continue
                        
                # Switch back to main content
                driver.switch_to.default_content()
                
            except:
                # If anything fails, switch back to main content and continue
                driver.switch_to.default_content()
                continue
                
        return "Could not click reCAPTCHA checkbox"
        
    except Exception as e:
        # Make sure we switch back to main content
        try:
            driver.switch_to.default_content()
        except:
            pass
        return f"Error handling reCAPTCHA: {str(e)}"
    
@tool
def combine_answer(fairprice_result: str = None, lazada_result: str = None) -> str:
    """
    Returns the final answer combining results from both websites.
    Args:
        fairprice_result: JSON string from FairPrice (optional)
        lazada_result: JSON string from Lazada (optional)
    Returns:
        str: Combined JSON response
    """
    import json
    
    try:
        # Parse the JSON strings
        fairprice_data = json.loads(fairprice_result) if fairprice_result else {}
        lazada_data = json.loads(lazada_result) if lazada_result else {}
        
        # Create combined response
        combined_result = {
            "fairprice": {
                "product": fairprice_data.get("product", "Not found"),
                "currentPrice": fairprice_data.get("currentPrice", "Not available"),
                "originalPrice": fairprice_data.get("originalPrice"),
                "promotion": fairprice_data.get("promotion")
            },
            "lazada": {
                "product": lazada_data.get("product", "Not found"),
                "currentPrice": lazada_data.get("currentPrice", "Not available"),
                "originalPrice": lazada_data.get("originalPrice"),
                "promotion": lazada_data.get("promotion")
            }
        }
        
        return f"Final combined results: {json.dumps(combined_result, indent=2)}"
    except Exception as e:
        return f"Error combining results: {str(e)}\nFairPrice raw: {fairprice_result}\nLazada raw: {lazada_result}"

agent = CodeAgent(
    tools=[go_back, close_popups, search_item_ctrl_f, input_search, click_product_image, get_product_details,handle_recaptcha,final_answer,combine_answer],
    model=model,
    additional_authorized_imports=["helium"],
    step_callbacks=[save_screenshot],
    max_steps=20,
    verbosity_level=2,
)

helium_instructions = """
You can use helium to access websites. Don't bother about the helium driver, it's already managed.
First you need to import everything from helium, then you can do other actions!
Code:
```py
from helium import *
go_to('github.com/trending')
```<end_code>

You can directly click clickable elements by inputting the text that appears on them.
Code:
```py
click("Top products")
```<end_code>

If it's a link:
Code:
```py
click(Link("Top products"))
```<end_code>

If you try to interact with an element and it's not found, you'll get a LookupError.
In general stop your action after each button click to see what happens on your screenshot.
Never try to login in a page.

To scroll up or down, use scroll_down or scroll_up with as an argument the number of pixels to scroll from.
Code:
```py
scroll_down(num_pixels=1) # This will scroll one viewport down
```<end_code>

When you have pop-ups with a cross icon to close, don't try to click the close icon by finding its element or targeting an 'X' element (this most often fails).
Just use your built-in tool `close_popups` to close them:
Code:
```py
close_popups()
```<end_code>

You can use .exists() to check for the existence of an element. For example:
Code:
```py
if Text('Accept cookies?').exists():
    click('I accept')
```<end_code>

When you encounter a reCAPTCHA verification, use the handle_recaptcha tool to attempt clicking the checkbox:
Code:
```py
handle_recaptcha()  # This will find and click the reCAPTCHA checkbox if it appears
```<end_code>
It's good practice to call handle_recaptcha after navigation or search actions that might trigger verification.

Proceed in several steps rather than trying to solve the task in one shot.
And at the end, only when you have your answer, return your final answer.
Code:
```py
final_answer("YOUR_ANSWER_HERE")
```<end_code>

If pages seem stuck on loading, you might have to wait, for instance `import time` and run `time.sleep(5.0)`. But don't overuse this!
To list elements on page, DO NOT try code-based element searches like 'contributors = find_all(S("ol > li"))': just look at the latest screenshot you have and read it visually, or use your tool search_item_ctrl_f.
Of course, you can act on buttons like a user would do when navigating.
After each code blob you write, you will be automatically provided with an updated screenshot of the browser and the current browser url.
But beware that the screenshot will only be taken at the end of the whole action, it won't see intermediate states.
Don't kill the browser.
"""
helium_instructions = helium_instructions + """
You can use input_search to type text into a search box and optionally submit the search:
Code:
```py
input_search("your search text")  # Will submit the search
input_search("your search text", submit=False)  # Will only input text without submitting
```<end_code>
"""

# Run the agent!
def combine_results(fairprice_result: str, lazada_result: str) -> str:
    """
    Combines results from FairPrice and Lazada into a single JSON response.
    Args:
        fairprice_result: JSON string from FairPrice
        lazada_result: JSON string from Lazada
    Returns:
        str: Combined JSON response
    """
    import json
    
    try:
        # Parse the JSON strings
        fairprice_data = json.loads(fairprice_result) if fairprice_result else {}
        lazada_data = json.loads(lazada_result) if lazada_result else {}
        
        # Create combined response
        combined_result = {
            "fairprice": {
                "product": fairprice_data.get("product", "Not found"),
                "currentPrice": fairprice_data.get("currentPrice", "Not available"),
                "originalPrice": fairprice_data.get("originalPrice"),
                "promotion": fairprice_data.get("promotion")
            },
            "lazada": {
                "product": lazada_data.get("product", "Not found"),
                "currentPrice": lazada_data.get("currentPrice", "Not available"),
                "originalPrice": lazada_data.get("originalPrice"),
                "promotion": lazada_data.get("promotion")
            }
        }
        
        return json.dumps(combined_result, indent=2)
    except Exception as e:
        return json.dumps({
            "error": f"Failed to combine results: {str(e)}",
            "fairprice_raw": fairprice_result,
            "lazada_raw": lazada_result
        }, indent=2)

# Modified search request that searches both websites
multi_site_search_request = """
I need you to do the following steps sequentially:
1. First import helium:
```py
from helium import *
```

2. Navigate to FairPrice website and search:
```py
go_to('https://fairprice.com.sg/')
```
3. Use input_search to search for the product
4. Click on the product image
5. Store the FairPrice result by running get_product_details()
6. Save this result to use later

7. Navigate to Lazada website and search:
```py
go_to('https://www.lazada.sg/')
```
8. Use input_search to search for the same product
9. Click on the product image
10. Get the Lazada result by running get_product_details()

11. Use the combine_answer tool with both stored results:
```python
combine_answer(fairprice_result, lazada_result)
```
"""

def run_multi_site_search(product_name: str):
    try:
        # Initialize driver before running the search
        global driver
        driver = initialize_driver()

        # Create agent with newly initialized driver
        agent = CodeAgent(
            tools=[go_back, close_popups, search_item_ctrl_f, input_search, 
                  click_product_image, get_product_details, handle_recaptcha,
                  final_answer, combine_answer],
            model=model,
            additional_authorized_imports=["helium"],
            step_callbacks=[save_screenshot],
            max_steps=20,
            verbosity_level=2,
        )
        
        # Create a custom search request with the product name
        custom_request = multi_site_search_request.replace("the product", f'"{product_name}"')
        
        # Run the agent
        response = agent.run(custom_request + helium_instructions)
        
        # Process agent logs to extract the final combined result
        final_combined_result = None
        
        # Process agent logs to extract results
        for step in agent.logs:
            if hasattr(step, 'action_output') and isinstance(step.action_output, str):
                # Check if this contains the combined result format
                if 'Final combined results:' in step.action_output:
                    # Extract the JSON part from the string
                    json_str = step.action_output.split('Final combined results:', 1)[1].strip()
                    if json_str.startswith('{') and json_str.endswith('}'):
                        final_combined_result = json_str
                        break

        if final_combined_result:
            return final_combined_result
        else:
            return None
            
    finally:
        # Clean up: close the browser
        try:
            helium.kill_browser()
        except:
            pass

# Streamlit UI
st.set_page_config(page_title="Price Comparison", layout="wide")

# Title and description
st.title("🛍️ Singapore Price Comparison")
st.markdown("Compare prices between FairPrice and Lazada")

# Input field for product name
product_name = st.text_input("Enter product name to search:", "iPhone 16 Pro Max")

# Search button
if st.button("Compare Prices"):
    if product_name:
        with st.spinner(f'Searching for "{product_name}" across stores...'):
            try:
                # Run the search
                result = run_multi_site_search(product_name)
                
                if result:
                    # Parse the JSON result
                    data = json.loads(result)
                    
                    # Create two columns for the comparison
                    col1, col2 = st.columns(2)
                    
                    # FairPrice Column
                    with col1:
                        st.subheader("🏪 FairPrice")
                        fairprice = data["fairprice"]
                        st.markdown(f"**Product:** {fairprice['product']}")
                        st.markdown(f"**Current Price:** {fairprice['currentPrice']}")
                        if fairprice['originalPrice']:
                            st.markdown(f"**Original Price:** {fairprice['originalPrice']}")
                        if fairprice['promotion']:
                            st.markdown(f"**Savings:** {fairprice['promotion']}")
                        else:
                            st.markdown("**Promotion:** No current promotions")
                    
                    # Lazada Column
                    with col2:
                        st.subheader("🛒 Lazada")
                        lazada = data["lazada"]
                        st.markdown(f"**Product:** {lazada['product']}")
                        st.markdown(f"**Current Price:** {lazada['currentPrice']}")
                        if lazada['originalPrice']:
                            st.markdown(f"**Original Price:** {lazada['originalPrice']}")
                        if lazada['promotion']:
                            st.markdown(f"**Savings:** {lazada['promotion']}")
                        else:
                            st.markdown("**Promotion:** No current promotions")
                    
                    # Price comparison
                    fp_price = float(fairprice['currentPrice'].replace('$', '').replace(',', ''))
                    lz_price = float(lazada['currentPrice'].replace('$', '').replace(',', ''))
                    price_diff = abs(fp_price - lz_price)
                    
                    st.markdown("---")
                    st.subheader("💰 Price Comparison")
                    
                    if fp_price > lz_price:
                        st.markdown(f"**Lazada** is **${price_diff:.2f}** cheaper than FairPrice")
                    elif lz_price > fp_price:
                        st.markdown(f"**FairPrice** is **${price_diff:.2f}** cheaper than Lazada")
                    else:
                        st.markdown("Both stores have the same price!")
                    
                    # Display raw JSON with formatting
                    with st.expander("Show Raw JSON"):
                        st.json(data)
                else:
                    st.error("No results found. Please try a different product name.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter a product name to search.")