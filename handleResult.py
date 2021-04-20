from gtts import gTTS
import os
from playsound import playsound

def text2Speach(mytext):
    # This function reads aloud the text that is inputed into it!!!
    
    language = 'en'
    myobj = gTTS(text=mytext, lang=language, slow=False)
    myobj.save("welcome.mp3")
  
    # Playing the converted file
    playsound("Welcome.mp3")