############################################# IMPORTING ################################################
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2,os
import csv
import numpy as np
import pandas as pd
from PIL import Image
import datetime
import time
import threading
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai


############################################# FUNCTIONS (From First Code) ################################################

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

##################################################################################

def tick():
    # This function is used by the new GUI's clock
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200,tick)

###################################################################################

def contact():
    mess._show(title='Contact us', message="Please contact us on : 'jeevang1405@gmail.com' ")

###################################################################################

def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if exists:
        pass
    else:
        mess._show(title='Some file missing', message='haarcascade_frontalface_default.xml is missing. Please contact us for help')
        window.destroy()

###################################################################################

# Using the improved change_pass GUI from the second code for better consistency
def save_pass():
    assure_path_exists("TrainingImageLabel/")
    try:
        with open("TrainingImageLabel\psd.txt", "r") as tf:
            key = tf.read()
    except FileNotFoundError:
        master.destroy()
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas:
            with open("TrainingImageLabel\psd.txt", "w") as tf:
                tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!')
        else:
            mess._show(title='No Password Entered', message='Password not set. Please try again.')
        return

    op = old.get()
    newp = new.get()
    nnewp = nnew.get()

    if op == key:
        if newp and newp == nnewp:
            with open("TrainingImageLabel\psd.txt", "w") as txf:
                txf.write(newp)
            mess._show(title='Password Changed', message='Password changed successfully!')
            master.destroy()
        else:
            mess._show(title='Error', message='New passwords do not match or are empty. Please try again.')
    else:
        mess._show(title='Wrong Password', message='Please enter the correct old password.')


def change_pass():
    global master, old, new, nnew
    master = tk.Toplevel(window)
    master.grab_set()
    master.title("Change Password")
    master.geometry("400x180")
    master.resizable(False, False)
    master.configure(background="#ffffff")

    tk.Label(master, text='Enter Old Password', bg='white', font=('Segoe UI', 12)).place(x=20, y=20)
    old = tk.Entry(master, width=22, fg="black", font=('Segoe UI', 12), show='*')
    old.place(x=190, y=20)

    tk.Label(master, text='Enter New Password', bg='white', font=('Segoe UI', 12)).place(x=20, y=55)
    new = tk.Entry(master, width=22, fg="black", font=('Segoe UI', 12), show='*')
    new.place(x=190, y=55)

    tk.Label(master, text='Confirm New Password', bg='white', font=('Segoe UI', 12)).place(x=20, y=90)
    nnew = tk.Entry(master, width=22, fg="black", font=('Segoe UI', 12), show='*')
    nnew.place(x=190, y=90)

    save_btn = tk.Button(master, text="Save", command=save_pass, fg="white", bg="#27ae60", font=('Segoe UI', 11, 'bold'), width=15)
    save_btn.place(x=40, y=130)
    cancel_btn = tk.Button(master, text="Cancel", command=master.destroy, fg="white", bg="#c0392b", font=('Segoe UI', 11, 'bold'), width=15)
    cancel_btn.place(x=210, y=130)

#####################################################################################

def psw():
    assure_path_exists("TrainingImageLabel/")
    exists1 = os.path.isfile("TrainingImageLabel\psd.txt")
    if exists1:
        tf = open("TrainingImageLabel\psd.txt", "r")
        key = tf.read()
    else:
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas == None:
            mess._show(title='No Password Entered', message='Password not set!! Please try again')
        else:
            tf = open("TrainingImageLabel\psd.txt", "w")
            tf.write(new_pas)
            mess._show(title='Password Registered', message='New password was registered successfully!!')
            return
    password = tsd.askstring('Password', 'Enter Password', show='*')
    if (password == key):
        TrainImages()
    elif (password == None):
        pass
    else:
        mess._show(title='Wrong Password', message='You have entered wrong password')

######################################################################################

def clear():
    txt.delete(0, 'end')
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)

def clear2():
    txt2.delete(0, 'end')
    res = "1)Take Images  >>>  2)Save Profile"
    message1.configure(text=res)

#######################################################################################

def TakeImages():
    check_haarcascadefile()
    columns = ['SERIAL NO.', '', 'ID', '', 'NAME']
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    serial = 0
    exists = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists:
        with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                serial = serial + 1
        serial = (serial // 2)
        csvFile1.close()
    else:
        with open("StudentDetails\StudentDetails.csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(columns)
            serial = 1
        csvFile1.close()
    Id = (txt.get())
    name = (txt2.get())
    if ((name.isalpha()) or (' ' in name)):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum = sampleNum + 1
                # Path and naming convention from First Code
                cv2.imwrite("TrainingImage\ " + name + "." + str(serial) + "." + Id + '.' + str(sampleNum) + ".jpg",
                            gray[y:y + h, x:x + w])
                cv2.imshow('Taking Images', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 100:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Taken for ID : " + Id
        row = [serial, '', Id, '', name]
        with open('StudentDetails\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message1.configure(text=res)
    else:
        # The new UI has 'message1' for status, not 'message'
        if (name.isalpha() == False):
            res = "Enter Correct name"
            message1.configure(text=res)

########################################################################################

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, ID = getImagesAndLabels("TrainingImage")
    try:
        recognizer.train(faces, np.array(ID))
    except:
        mess._show(title='No Registrations', message='Please Register someone first!!!')
        return
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Profile Saved Successfully"
    # Update the status label (message1) and registration count label
    message1.configure(text=res)
    update_registration_count()
    mess._show(title="Success", message="Profile saved successfully!")


############################################################################################3

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        # ID extraction from First Code's naming convention
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids

###########################################################################################

def TrackImages():
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    assure_path_exists("StudentDetails/")
    for k in tv.get_children():
        tv.delete(k)
    msg = ''
    i = 0
    j = 0
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    exists3 = os.path.isfile("TrainingImageLabel\Trainner.yml")
    if exists3:
        recognizer.read("TrainingImageLabel\Trainner.yml")
    else:
        mess._show(title='Data Missing', message='Please click on Save Profile to reset data!!')
        return
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', '', 'Name', '', 'Date', '', 'Time']
    exists1 = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists1:
        df = pd.read_csv("StudentDetails\StudentDetails.csv")
    else:
        mess._show(title='Details Missing', message='Students details are missing, please check!')
        cam.release()
        cv2.destroyAllWindows()
        window.destroy()
        return # Added return to stop execution
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if (conf < 50):
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['SERIAL NO.'] == serial]['NAME'].values
                ID = df.loc[df['SERIAL NO.'] == serial]['ID'].values
                ID = str(ID)
                ID = ID[1:-1]
                bb = str(aa)
                bb = bb[2:-2]
                attendance = [str(ID), '', bb, '', str(date), '', str(timeStamp)]
            else:
                Id = 'Unknown'
                bb = str(Id)
            cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)
        cv2.imshow('Taking Attendance', im)
        if (cv2.waitKey(1) == ord('q')):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    exists = os.path.isfile("Attendance\Attendance_" + date + ".csv")
    if exists:
        with open("Attendance\Attendance_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(attendance)
        csvFile1.close()
    else:
        with open("Attendance\Attendance_" + date + ".csv", 'a+') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(col_names)
            writer.writerow(attendance)
        csvFile1.close()
    with open("Attendance\Attendance_" + date + ".csv", 'r') as csvFile1:
        reader1 = csv.reader(csvFile1)
        for lines in reader1:
            i = i + 1
            if (i > 1):
                if (i % 2 != 0):
                    iidd = str(lines[0]) + '   '
                    # The new treeview has different column identifiers
                    tv.insert('', 'end', text=iidd, values=(str(lines[2]), str(lines[4]), str(lines[6])))
    csvFile1.close()
    cam.release()
    cv2.destroyAllWindows()

def update_registration_count():
    # Helper function to update the registration count on the new GUI
    res=0
    exists = os.path.isfile("StudentDetails\StudentDetails.csv")
    if exists:
        with open("StudentDetails\StudentDetails.csv", 'r') as csvFile1:
            reader1 = csv.reader(csvFile1)
            for l in reader1:
                res = res + 1
        # The logic from the first code to calculate registrations
        res = (res // 2) - 1 if res > 1 else 0
        csvFile1.close()
    else:
        res = 0
    reg_count_label.config(text=f"Total Registrations: {res}")


######################################## GUI FRONT-END (REDESIGNED) ###########################################

# --- Color Scheme & Fonts ---
BG_COLOR = "#1F2833"
FRAME_COLOR = "#2C3E50"
TEXT_COLOR = "#FFFFFF"
ACCENT_COLOR = "#66FCF1"
ALT_ACCENT_COLOR = "#45A29E"
FONT_BOLD = ("Segoe UI", 12, "bold")
FONT_NORMAL = ("Segoe UI", 15)
FONT_TITLE = ("Segoe UI", 32, "bold")
FONT_HEADER = ("Segoe UI", 16, "bold")

# --- Main Window Setup ---
window = tk.Tk()
window.geometry("1280x720")
window.title("Aura Attend")
window.configure(bg=BG_COLOR)

# --- Header Frame ---
header_frame = tk.Frame(window, bg=FRAME_COLOR, pady=10)
header_frame.pack(fill='x')

title_label = tk.Label(header_frame, text=" AURA  ATTEND", fg=ACCENT_COLOR, bg=FRAME_COLOR, font=FONT_TITLE)
title_label.pack()

# Create a canvas for the circular assistant button
assistant_canvas = tk.Canvas(header_frame, width=60, height=60, bg=FRAME_COLOR, bd=0, highlightthickness=0, cursor="hand2")
assistant_canvas.place(relx=0.98, rely=0.5, anchor='e') # Place on the right edge, vertically centered

# Draw the red circle and text
circle = assistant_canvas.create_oval(5, 5, 55, 55, fill="#c0392b", outline="")
assistant_text = assistant_canvas.create_text(30, 30, text="AI", fill="white", font=("Segoe UI", 16, "bold"))

# Bind click event to both the circle and the text
assistant_canvas.tag_bind(circle, '<Button-1>', lambda e: assistant(window))
assistant_canvas.tag_bind(assistant_text, '<Button-1>', lambda e: assistant(window))

# Define hover effects
def on_assistant_enter(e):
    assistant_canvas.itemconfig(circle, fill="#E74C3C") # Lighter red
def on_assistant_leave(e):
    assistant_canvas.itemconfig(circle, fill="#c0392b") # Original red

# Bind hover events to both the circle and the text
assistant_canvas.tag_bind(circle, '<Enter>', on_assistant_enter)
assistant_canvas.tag_bind(circle, '<Leave>', on_assistant_leave)
assistant_canvas.tag_bind(assistant_text, '<Enter>', on_assistant_enter)
assistant_canvas.tag_bind(assistant_text, '<Leave>', on_assistant_leave)


time_frame = tk.Frame(header_frame, bg=FRAME_COLOR)
time_frame.pack(pady=5)

ts = time.time()
date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
mont={'01':'January','02':'February','03':'March','04':'April','05':'May','06':'June','07':'July','08':'August','09':'September','10':'October','11':'November','12':'December'}
day,month,year=date.split("-")

date_label = tk.Label(time_frame, text=f"{day} {mont[month]} {year}", fg=TEXT_COLOR, bg=FRAME_COLOR, font=FONT_NORMAL)
date_label.pack(side='left', padx=10)

clock = tk.Label(time_frame, fg=TEXT_COLOR, bg=FRAME_COLOR, font=FONT_NORMAL)
clock.pack(side='left', padx=10)
tick() # Original tick function call

# --- Main Content Frame ---
main_frame = tk.Frame(window, bg=BG_COLOR)
main_frame.pack(pady=20, padx=20, fill='both', expand=True)
main_frame.columnconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(0, weight=1)

# --- Registration Frame (Right Side) ---
reg_frame = tk.Frame(main_frame, bg=FRAME_COLOR, bd=2, relief='groove')
reg_frame.grid(row=0, column=1, sticky="nsew", padx=10)

reg_header = tk.Label(reg_frame, text="New Registration", font=FONT_HEADER, fg=ACCENT_COLOR, bg=FRAME_COLOR)
reg_header.pack(pady=20)

tk.Label(reg_frame, text="Enter Registration No.", font=FONT_NORMAL, fg=TEXT_COLOR, bg=FRAME_COLOR).pack(pady=(10,0))
# txt is the variable name from First Code
txt = tk.Entry(reg_frame, width=30, font=FONT_NORMAL, bg="#34495E", fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
txt.pack(pady=5, padx=20)

tk.Label(reg_frame, text="Enter Full Name", font=FONT_NORMAL, fg=TEXT_COLOR, bg=FRAME_COLOR).pack(pady=(10,0))
# txt2 is the variable name from First Code
txt2 = tk.Entry(reg_frame, width=30, font=FONT_NORMAL, bg="#34495E", fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
txt2.pack(pady=5, padx=20)

clear_frame = tk.Frame(reg_frame, bg=FRAME_COLOR)
clear_frame.pack(pady=5)
# Buttons call original clear/clear2 functions
clearButton = tk.Button(clear_frame, text="Clear ID", command=clear, font=FONT_NORMAL, width=12, bg="#E74C3C")
clearButton.pack(side='left', padx=5)
clearButton2 = tk.Button(clear_frame, text="Clear Name", command=clear2, font=FONT_NORMAL, width=12, bg="#E74C3C")
clearButton2.pack(side='left', padx=5)

# message1 is the variable name for the status label from First Code
message1 = tk.Label(reg_frame, text="1)Take Images  >>>  2)Save Profile", bg=FRAME_COLOR, fg=TEXT_COLOR, font=FONT_NORMAL)
message1.pack(pady=10)

reg_btn_frame = tk.Frame(reg_frame, bg=FRAME_COLOR)
reg_btn_frame.pack(expand=True, pady=15)

# Buttons call original TakeImages/psw (which calls TrainImages) functions
takeImg = tk.Button(reg_btn_frame, text="Take Images", command=TakeImages, font=FONT_BOLD, width=30, height=2, bg=ALT_ACCENT_COLOR, fg=TEXT_COLOR)
takeImg.pack(pady=10)
trainImg = tk.Button(reg_btn_frame, text="Save Profile", command=psw, font=FONT_BOLD, width=30, height=2, bg=ACCENT_COLOR, fg=BG_COLOR)
trainImg.pack(pady=10)

reg_count_label = tk.Label(reg_frame, text="", bg=FRAME_COLOR, fg=TEXT_COLOR, font=FONT_NORMAL)
reg_count_label.pack(side='bottom', pady=10)
update_registration_count() # Initial count check

# --- Attendance Frame (Left Side) ---
att_frame = tk.Frame(main_frame, bg=FRAME_COLOR, bd=2, relief='groove')
att_frame.grid(row=0, column=0, sticky="nsew", padx=10)

att_header = tk.Label(att_frame, text="Attendance Log", font=FONT_HEADER, fg=ACCENT_COLOR, bg=FRAME_COLOR)
att_header.pack(pady=20)

# --- TreeView Customization ---
style = ttk.Style()
style.theme_use("default")
style.configure("Treeview", background=FRAME_COLOR, foreground=TEXT_COLOR, fieldbackground=FRAME_COLOR, rowheight=25, font=FONT_NORMAL)
style.map('Treeview', background=[('selected', ALT_ACCENT_COLOR)])
style.configure("Treeview.Heading", background=BG_COLOR, foreground=ACCENT_COLOR, font=FONT_BOLD, padding=(10,10))
style.map("Treeview.Heading", background=[('active', ALT_ACCENT_COLOR)])

tree_frame = tk.Frame(att_frame, bg=FRAME_COLOR)
tree_frame.pack(expand=True, fill='both', padx=10)

scroll = ttk.Scrollbar(tree_frame, orient='vertical')
# tv is the variable name for the Treeview from First Code
tv = ttk.Treeview(tree_frame, height=13, columns=('name', 'date', 'time'), yscrollcommand=scroll.set)
scroll.config(command=tv.yview)

# Column setup from First Code
tv.column('#0',width=82)
tv.column('name',width=130)
tv.column('date',width=133)
tv.column('time',width=133)
tv.heading('#0',text ='ID')
tv.heading('name',text ='NAME')
tv.heading('date',text ='DATE')
tv.heading('time',text ='TIME')

tv.pack(side='left', fill='both', expand=True)
scroll.pack(side='right', fill='y')

att_btn_frame = tk.Frame(att_frame, bg=FRAME_COLOR)
att_btn_frame.pack(pady=15, fill='x', side='bottom')

# Buttons call original TrackImages/destroy functions
trackImg = tk.Button(att_btn_frame, text="Take Attendance", command=TrackImages, font=FONT_BOLD, height=2, bg=ACCENT_COLOR, fg=BG_COLOR)
trackImg.pack(side='left', expand=True, padx=10)
quitWindow = tk.Button(att_btn_frame, text="     Exit    ", command=window.destroy, font=FONT_BOLD, height=2, bg="#c0392b", fg=TEXT_COLOR)
quitWindow.pack(side='right', expand=True, padx=10)

##################### MENUBAR #################################
menubar = tk.Menu(window)
# Connecting the menu to original functions
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label='Change Password', command=change_pass)
filemenu.add_command(label='Contact Us', command=contact)
filemenu.add_separator()
filemenu.add_command(label='Exit',command = window.destroy)
menubar.add_cascade(label='Options', menu=filemenu)
window.config(menu=menubar)

####################### ASSISTANT (NOW WITH STOP BUTTON) ###############################
class AssistantUI:
    def __init__(self, parent_window):
        self.window = tk.Toplevel(parent_window)
        self.window.title("Voice Assistant")
        self.window.geometry("550x650")
        self.window.configure(bg=BG_COLOR)
        self.window.transient(parent_window)
        self.window.grab_set()

        # --- Initialize TTS Engine ---
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)
        
        # --- UI Elements ---
        self.status_label = tk.Label(self.window, text="Ready", font=FONT_NORMAL, fg=TEXT_COLOR, bg=BG_COLOR)
        self.status_label.pack(pady=10)

        text_frame = tk.Frame(self.window, bg=FRAME_COLOR, bd=1)
        text_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        self.conversation_text = tk.Text(text_frame, wrap=tk.WORD, bg=FRAME_COLOR, fg=TEXT_COLOR, font=("Segoe UI", 12), bd=0, padx=5, pady=5, insertbackground=TEXT_COLOR)
        self.conversation_text.pack(side="left", fill="both", expand=True)

        scrollbar = ttk.Scrollbar(text_frame, command=self.conversation_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.conversation_text.config(yscrollcommand=scrollbar.set)
        
        self.conversation_text.tag_config("user_tag", foreground=ACCENT_COLOR, font=FONT_BOLD)
        self.conversation_text.tag_config("assistant_tag", foreground=TEXT_COLOR, font=FONT_NORMAL)
        self.conversation_text.tag_config("error_tag", foreground="#E74C3C", font=FONT_NORMAL)
        self.conversation_text.config(state=tk.DISABLED)

        input_frame = tk.Frame(self.window, bg=BG_COLOR)
        input_frame.pack(pady=10, padx=10, fill='x')
        input_frame.columnconfigure(0, weight=1) # Make the entry column expandable
        
        self.text_entry = tk.Entry(input_frame, font=FONT_NORMAL, bg="#34495E", fg=TEXT_COLOR, insertbackground=TEXT_COLOR, width=40)
        self.text_entry.grid(row=0, column=0, sticky="ew", ipady=8, padx=(0,5))
        self.text_entry.bind("<Return>", self.start_text_query_thread)

        self.send_button = tk.Button(input_frame, text="Send", command=self.start_text_query_thread, font=FONT_BOLD, bg=ALT_ACCENT_COLOR, fg=TEXT_COLOR, width=6)
        self.send_button.grid(row=0, column=1, ipady=1)
        
        self.listen_button = tk.Button(input_frame, text="Listen", command=self.start_voice_query_thread, font=FONT_BOLD, bg=ACCENT_COLOR, fg=BG_COLOR, width=6)
        self.listen_button.grid(row=0, column=2, ipady=1, padx=5)
        
        self.stop_button = tk.Button(input_frame, text="Stop", command=self.stop_speaking, font=FONT_BOLD, bg="#E74C3C", fg=TEXT_COLOR, width=6, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=3, ipady=1)

    def _schedule_ui_update(self, func):
        if self.window.winfo_exists():
            self.window.after(0, func)
            
    def disable_inputs(self):
        self._schedule_ui_update(lambda: self.listen_button.config(state=tk.DISABLED, bg=ALT_ACCENT_COLOR))
        self._schedule_ui_update(lambda: self.send_button.config(state=tk.DISABLED, bg=ALT_ACCENT_COLOR))
        self._schedule_ui_update(lambda: self.text_entry.config(state=tk.DISABLED))

    def enable_inputs(self):
        self._schedule_ui_update(lambda: self.listen_button.config(state=tk.NORMAL, bg=ACCENT_COLOR))
        self._schedule_ui_update(lambda: self.send_button.config(state=tk.NORMAL, bg=ALT_ACCENT_COLOR))
        self._schedule_ui_update(lambda: self.text_entry.config(state=tk.NORMAL))
        self._schedule_ui_update(lambda: self.stop_button.config(state=tk.DISABLED))
        self.update_status("Ready")

    def start_voice_query_thread(self):
        self.disable_inputs()
        thread = threading.Thread(target=assistant_voice_task, args=(self,))
        thread.daemon = True
        thread.start()
        
    def start_text_query_thread(self, event=None):
        query = self.text_entry.get().strip()
        if not query:
            return
        self.text_entry.delete(0, tk.END)
        self.add_conversation_entry("You", query, "user_tag")
        self.disable_inputs()
        thread = threading.Thread(target=process_query, args=(query, self))
        thread.daemon = True
        thread.start()
        
    def speak(self, text):
        if text:
            try:
                self._schedule_ui_update(lambda: self.stop_button.config(state=tk.NORMAL))
                self.update_status("Speaking...")
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"Error during speech: {e}")
            finally:
                if self.window.winfo_exists():
                    self._schedule_ui_update(lambda: self.stop_button.config(state=tk.DISABLED))
                    if self.status_label.cget("text") == "Speaking...":
                        self.update_status("Ready")

    def stop_speaking(self):
        if self.engine._inLoop:
            self.engine.stop()

    def update_status(self, text):
        self._schedule_ui_update(lambda: self.status_label.config(text=text))

    def add_conversation_entry(self, who, text, tag):
        def _insert():
            self.conversation_text.config(state=tk.NORMAL)
            self.conversation_text.insert(tk.END, f"{who}: ", (tag,))
            self.conversation_text.insert(tk.END, f"{text}\n\n")
            self.conversation_text.config(state=tk.DISABLED)
            self.conversation_text.see(tk.END)
        self._schedule_ui_update(_insert)

# --- MODIFIED Data Handling Functions for the Assistant ---
def _get_all_attendance_data():
    """Reads all attendance CSVs into a single DataFrame, handling potential errors."""
    attendance_dir = "Attendance/"
    if not os.path.exists(attendance_dir):
        return None
    all_files = [os.path.join(attendance_dir, f) for f in os.listdir(attendance_dir) if f.endswith('.csv')]
    if not all_files:
        return None
    try:
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        df.drop_duplicates(subset=['Name', 'Date'], inplace=True)
        return df
    except (pd.errors.EmptyDataError, ValueError):
        return pd.DataFrame(columns=['Name', 'Date', 'Time'])

def _get_all_registered_students():
    """Returns a list of all students from the main details file."""
    try:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        return df['NAME'].unique().tolist()
    except (FileNotFoundError, KeyError):
        return None

def _get_todays_present_students():
    """Returns a list of students present today."""
    today = datetime.datetime.now().strftime('%d-%m-%Y')
    filepath = f"Attendance/Attendance_{today}.csv"
    try:
        df = pd.read_csv(filepath)
        return df['Name'].unique().tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return []

def handle_attendance_query(query: str):
    """
    Processes natural language queries about student attendance.
    Returns a string response or None if the query can't be handled locally.
    """
    query_lower = query.lower()
    
    # --- QUERIES ABOUT TODAY'S ATTENDANCE ---
    if "today" in query_lower or "currently" in query_lower or "right now" in query_lower:
        present_students = _get_todays_present_students()
        all_students = _get_all_registered_students()

        if all_students is None:
            return "I can't find the student details file. It seems no students are registered."
        if "how many" in query_lower and "present" in query_lower:
            return f"There are {len(present_students)} students present today."
        if "how many" in query_lower and "absent" in query_lower:
            absent_count = len(all_students) - len(present_students)
            return f"There are {absent_count} students absent today."
        if ("which students" in query_lower and "present" in query_lower) or "list of all present" in query_lower:
            if present_students:
                return f"The students present today are: {', '.join(sorted(present_students))}."
            else:
                return "No one has been marked present yet today."
        if ("which students" in query_lower and "absent" in query_lower) or "list of all absent" in query_lower:
            absent_students = sorted(list(set(all_students) - set(present_students)))
            if absent_students:
                 return f"The students absent today are: {', '.join(absent_students)}."
            elif not present_students:
                 return "Attendance has not been taken yet, so all students are currently marked absent."
            else:
                return "It looks like everyone is present today."

    # --- QUERIES ABOUT TOTAL REGISTERED STUDENTS ---
    if "how many students" in query_lower and ("registered" in query_lower or "enrolled" in query_lower or "total" in query_lower):
        all_students = _get_all_registered_students()
        if all_students:
            return f"There are a total of {len(all_students)} students registered in the system."
        else:
            return "I couldn't find the student details file. No students seem to be registered."

    # --- QUERIES FOR A SPECIFIC STUDENT ---
    if "attendance of" in query_lower or "attendance for" in query_lower or "how many days has" in query_lower:
        try:
            name_part = query_lower.split("of")[-1].split("for")[-1].split("has")[-1]
            student_name = name_part.replace("been present", "").strip().title()
            
            if not student_name: return None 
            full_df = _get_all_attendance_data()
            if full_df is None or full_df.empty:
                return "There is no attendance data available."

            student_records = full_df[full_df['Name'].str.title() == student_name]
            attendance_count = len(student_records)
            
            if attendance_count > 0:
                 if "dates" in query_lower:
                     dates = sorted(student_records['Date'].unique().tolist())
                     return f"{student_name} was present on the following {len(dates)} dates: {', '.join(dates)}."
                 else:
                     return f"{student_name} has been present for {attendance_count} days."
            else:
                all_registered = _get_all_registered_students()
                if all_registered and student_name in [s.title() for s in all_registered]:
                    return f"{student_name} is registered but has zero recorded attendance days."
                else:
                    return f"I couldn't find any attendance records for a student named {student_name}."
        except Exception:
            return "I couldn't quite understand that student's name. Please try again."

    # --- QUERIES ABOUT ATTENDANCE TRENDS AND RANKINGS ---
    if "most attendance" in query_lower or "highest attendance" in query_lower or "least attendance" in query_lower or "lowest attendance" in query_lower:
        full_df = _get_all_attendance_data()
        if full_df is None or full_df.empty:
            return "There is no attendance data available to compare."
        
        counts = full_df['Name'].value_counts()
        if "most" in query_lower or "highest" in query_lower:
            student = counts.idxmax(); count = counts.max()
            return f"{student} has the highest attendance with {count} days present."
        else:
            student = counts.idxmin(); count = counts.min()
            return f"{student} has the lowest attendance with {count} days present."
            
    # --- ADVANCED QUERIES ---
    if "trend" in query_lower and ("absent" in query_lower or "absenteeism" in query_lower):
        full_df = _get_all_attendance_data()
        all_students_list = _get_all_registered_students()
        if full_df is None or full_df.empty or not all_students_list:
            return "There isn't enough data to identify a trend."
        try:
            attendances_per_day = full_df.groupby('Date').size()
            absences_per_day = len(all_students_list) - attendances_per_day
            absences_per_day_df = absences_per_day.reset_index(name='absences')
            absences_per_day_df['Date'] = pd.to_datetime(absences_per_day_df['Date'], format='%d-%m-%Y')
            absences_per_day_df['DayOfWeek'] = absences_per_day_df['Date'].dt.day_name()
            avg_absences_per_day = absences_per_day_df.groupby('DayOfWeek')['absences'].mean()
            day_with_most_absences = avg_absences_per_day.idxmax()
            return f"Based on historical data, the trend shows the highest number of absences on an average {day_with_most_absences}."
        except Exception:
            return "I had trouble analyzing the trend. There might not be enough historical data."
            
    return None

# --- REFACTORED Core assistant logic ---
def process_query(query: str, ui: AssistantUI):
    try:
        ui.update_status("Thinking...")
        local_response = handle_attendance_query(query)
        
        if local_response:
            response_text = local_response
        else:
            # !!! IMPORTANT !!!
            # You MUST replace "YOUR_GEMINI_API_KEY" with your actual Google AI API key.
            genai.configure(api_key="AIzaSyCqGzWekUoXKgZMLvpSQ3dr2unKx-D1gxA") 
            model = genai.GenerativeModel('gemma-3n-e2b-it')
            response = model.generate_content(query)
            response_text = ''.join(part.text for part in response.parts) if response.parts else "I'm sorry, I couldn't find an answer for that."
        
        ui.add_conversation_entry("Assistant", response_text, "assistant_tag")
        ui.speak(response_text)

    except ImportError:
        ui.add_conversation_entry("System Error", "A required library is missing. Please check your installation.", "error_tag")
    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        print(f"ASSISTANT ERROR: {error_message}")
        ui.add_conversation_entry("System Error", error_message, "error_tag")
    finally:
        if ui.window.winfo_exists():
            ui.enable_inputs()

def assistant_voice_task(ui: AssistantUI):
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            ui.speak("How can I help you?")
            
            if not ui.window.winfo_exists(): return
            
            ui.update_status("Listening...")
            try:
                audio = r.listen(source, timeout=7, phrase_time_limit=15)
            except sr.WaitTimeoutError:
                ui.update_status("Listening timed out.")
                ui.enable_inputs()
                return

        ui.update_status("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        ui.add_conversation_entry("You", query, "user_tag")
        process_query(query, ui)

    except sr.UnknownValueError:
        msg = "My apologies, I couldn't understand that."
        ui.add_conversation_entry("Assistant", msg, "error_tag")
        if ui.window.winfo_exists(): ui.enable_inputs()
    except sr.RequestError:
        msg = "I'm having trouble with my connection right now."
        ui.add_conversation_entry("Assistant", msg, "error_tag")
        if ui.window.winfo_exists(): ui.enable_inputs()
    except Exception as e:
        error_message = f"An unexpected error occurred during voice input: {e}"
        print(f"VOICE TASK ERROR: {error_message}")
        ui.add_conversation_entry("System Error", error_message, "error_tag")
        if ui.window.winfo_exists(): ui.enable_inputs()

# Main function to launch the assistant UI
def assistant(parent_window):
    AssistantUI(parent_window)

##################### END ######################################
window.mainloop()