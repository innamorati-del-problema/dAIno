from selenium import webdriver
import random
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from torch import long
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep
import cv2
import time


driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()))

driver.set_window_size(900, 600)
try:
    driver.get("chrome:dino")
except:
    pass

# dino perform a short jump


def short_jump(driver):
    webdriver.ActionChains(driver).key_down(Keys.SPACE).perform()
    sleep(0.15)
    webdriver.ActionChains(driver).key_up(Keys.SPACE).perform()

# dino perform a long jump


def long_jump(driver):
    webdriver.ActionChains(driver).key_down(Keys.SPACE).perform()
    sleep(1)
    webdriver.ActionChains(driver).key_up(Keys.SPACE).perform()

# dino ducks


def duck(driver):
    webdriver.ActionChains(driver).key_down(Keys.ARROW_DOWN).perform()
    sleep(1)
    webdriver.ActionChains(driver).key_up(Keys.ARROW_DOWN).perform()


driver.execute_script(" Runner.instance_.gameOver = function(){}")

driver.execute_script(
    " Runner.config.INVERT_DISTANCE = 9999999999999999999999999999")

driver.execute_script("Runner.instance_.tRex.setJumpVelocity(8.5)")


while True:
    # n = random.randint(1, 3)
    # print(n)
    # if n == 1:
    #     short_jump(driver)
    # elif n == 2:
    #     long_jump(driver)
    # else:
    #     duck(driver)
    driver.execute_script("console.log(Runner.instance_)")

    sleep(0.8)
