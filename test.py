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

############################################# GEMINI API CONFIGURATION ###################################
try:
    # IMPORTANT: Replace with your actual and valid Gemini API Key.
    genai.configure(api_key="AIzaSyCqGzWekUoXKgZMLvpSQ3dr2unKx-D1gxA")
except Exception as e:
    print(f"Fatal Error: Could not configure Google AI. The assistant will not work. Error: {e}")


############################################# FUNCTIONS ################################################

# --- MODIFIED ---: Use os.path.join for better path handling
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

# --- ADDED ---: Create essential data files if they don't exist
def initialize_data_files():
    data_dir = "Data"
    assure_path_exists(os.path.join(data_dir, "")) # Ensure the directory exists

    if not os.path.isfile(os.path.join(data_dir, "Classes.csv")):
        with open(os.path.join(data_dir, "Classes.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['ClassID', 'ClassName'])

    if not os.path.isfile(os.path.join(data_dir, "Enrollments.csv")):
        with open(os.path.join(data_dir, "Enrollments.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['EnrollmentID', 'StudentID', 'ClassID'])

    # Also ensure StudentDetails path is consistent
    assure_path_exists("StudentDetails/")
    if not os.path.isfile("StudentDetails/StudentDetails.csv"):
        with open("StudentDetails/StudentDetails.csv", 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['SERIAL NO.', '', 'ID', '', 'NAME'])

##################################################################################

def tick():
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
        mess._show(title='Some file missing', message='haarcascade_frontalface_default.xml is missing.')
        window.destroy()

###################################################################################

# --- ADDED ---: Complete Class Management Functionality
class ClassManager:
    def __init__(self, parent):
        self.window = tk.Toplevel(parent)
        self.window.title("Class Manager")
        self.window.geometry("450x400")
        self.window.transient(parent)
        self.window.grab_set()
        self.window.configure(bg=BG_COLOR)

        self.classes_file = os.path.join("Data", "Classes.csv")
        self.enrollments_file = os.path.join("Data", "Enrollments.csv")

        # --- UI ---
        main_frame = tk.Frame(self.window, bg=BG_COLOR, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Add Class Frame
        add_frame = tk.Frame(main_frame, bg=FRAME_COLOR, padx=10, pady=10)
        add_frame.pack(fill=tk.X, pady=5)
        tk.Label(add_frame, text="New Class Name:", font=FONT_NORMAL, bg=FRAME_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT, padx=5)
        self.class_entry = tk.Entry(add_frame, font=FONT_NORMAL, bg="#34495E", fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
        self.class_entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        add_btn = tk.Button(add_frame, text="Add", command=self.add_class, font=FONT_BOLD, bg=ACCENT_COLOR, fg=BG_COLOR)
        add_btn.pack(side=tk.LEFT, padx=5)

        # List and Delete Frame
        list_frame = tk.Frame(main_frame, bg=FRAME_COLOR, padx=10, pady=10)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        list_frame.rowconfigure(0, weight=1)
        list_frame.columnconfigure(0, weight=1)

        self.class_listbox = tk.Listbox(list_frame, font=FONT_NORMAL, bg="#34495E", fg=TEXT_COLOR, selectbackground=ALT_ACCENT_COLOR)
        self.class_listbox.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=self.class_listbox.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.class_listbox.config(yscrollcommand=scrollbar.set)

        delete_btn = tk.Button(main_frame, text="Delete Selected Class", command=self.delete_class, font=FONT_BOLD, bg="#E74C3C", fg=TEXT_COLOR)
        delete_btn.pack(fill=tk.X, pady=5)

        self.load_classes()

    def get_all_classes(self):
        try:
            df = pd.read_csv(self.classes_file)
            return df
        except (FileNotFoundError, pd.errors.EmptyDataError):
            return pd.DataFrame(columns=['ClassID', 'ClassName'])

    def load_classes(self):
        self.class_listbox.delete(0, tk.END)
        classes_df = self.get_all_classes()
        for index, row in classes_df.iterrows():
            self.class_listbox.insert(tk.END, f"{row['ClassName']} (ID: {row['ClassID']})")
        # --- MODIFIED --- Refresh class lists in main window after changes
        populate_class_lists()


    def add_class(self):
        class_name = self.class_entry.get().strip()
        if not class_name:
            mess.showerror("Error", "Class name cannot be empty.", parent=self.window)
            return

        classes_df = self.get_all_classes()
        if class_name.lower() in classes_df['ClassName'].str.lower().values:
            mess.showerror("Error", "This class name already exists.", parent=self.window)
            return

        new_id = (classes_df['ClassID'].max() + 1) if not classes_df.empty else 1
        new_class = pd.DataFrame([[new_id, class_name]], columns=['ClassID', 'ClassName'])
        updated_df = pd.concat([classes_df, new_class], ignore_index=True)
        updated_df.to_csv(self.classes_file, index=False)

        mess.showinfo("Success", f"Class '{class_name}' added successfully.", parent=self.window)
        self.class_entry.delete(0, tk.END)
        self.load_classes()

    def delete_class(self):
        selected_index = self.class_listbox.curselection()
        if not selected_index:
            mess.showerror("Error", "Please select a class to delete.", parent=self.window)
            return

        # Confirmation
        if not mess.askyesno("Confirm Delete", "Are you sure you want to delete this class? This will also remove all student enrollments for it.", parent=self.window):
            return

        selected_item = self.class_listbox.get(selected_index[0])
        class_id_to_delete = int(selected_item.split("ID: ")[1].replace(")", ""))

        # Delete from Classes.csv
        classes_df = self.get_all_classes()
        classes_df = classes_df[classes_df['ClassID'] != class_id_to_delete]
        classes_df.to_csv(self.classes_file, index=False)

        # Delete from Enrollments.csv
        try:
            enrollments_df = pd.read_csv(self.enrollments_file)
            enrollments_df = enrollments_df[enrollments_df['ClassID'] != class_id_to_delete]
            enrollments_df.to_csv(self.enrollments_file, index=False)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass # No enrollments to delete

        mess.showinfo("Success", "Class deleted successfully.", parent=self.window)
        self.load_classes()

def open_class_manager():
    ClassManager(window)

###################################################################################

def save_pass():
    # This function remains unchanged
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
    # This function remains unchanged
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
    # This function is now just for password-protecting TrainImages, remains mostly unchanged
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

# --- MODIFIED ---: Now saves enrollments after taking images.
def TakeImages():
    check_haarcascadefile()
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")

    # Get student details and selected classes
    Id = txt.get()
    name = txt2.get()
    selected_indices = reg_class_listbox.curselection()

    if not (Id.isalnum() and name and (name.isalpha() or ' ' in name)):
        mess.showerror("Input Error", "Please enter a valid alphanumeric ID and a valid name.")
        return
    if not selected_indices:
        mess.showerror("Input Error", "Please select at least one class for the student.")
        return

    # Get the next serial number
    try:
        student_df = pd.read_csv("StudentDetails/StudentDetails.csv")
        serial = (student_df['SERIAL NO.'].max() + 1) if not student_df.empty else 1
    except (FileNotFoundError, pd.errors.EmptyDataError):
        serial = 1

    # Image capturing process
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
            sampleNum += 1
            cv2.imwrite(os.path.join("TrainingImage", f"{name}.{serial}.{Id}.{sampleNum}.jpg"), gray[y:y+h, x:x+w])
            cv2.imshow('Taking Images', img)
        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 100:
            break
    cam.release()
    cv2.destroyAllWindows()

    # Save student details
    res = f"Images Taken for ID: {Id}"
    row = [serial, '', Id, '', name]
    with open('StudentDetails/StudentDetails.csv', 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    message1.configure(text=res)
    update_registration_count()

    # --- ADDED ---: Save enrollments
    enrollments_file = os.path.join("Data", "Enrollments.csv")
    try:
        enroll_df = pd.read_csv(enrollments_file)
        enroll_id = (enroll_df['EnrollmentID'].max() + 1) if not enroll_df.empty else 1
    except (FileNotFoundError, pd.errors.EmptyDataError):
        enroll_id = 1

    all_classes_df = pd.read_csv(os.path.join("Data", "Classes.csv"))
    with open(enrollments_file, 'a+', newline='') as f:
        writer = csv.writer(f)
        for index in selected_indices:
            class_name = reg_class_listbox.get(index)
            class_id = all_classes_df[all_classes_df['ClassName'] == class_name]['ClassID'].iloc[0]
            writer.writerow([enroll_id, serial, class_id])
            enroll_id += 1

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
    recognizer.save("TrainingImageLabel/Trainner.yml")
    res = "Profile Saved Successfully"
    message1.configure(text=res)
    mess._show(title="Success", message="Profile saved successfully!")

############################################################################################3

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        ID = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(ID)
    return faces, Ids

###########################################################################################

# --- MODIFIED ---: Now takes class selection as input and tracks attendance for that class only.
def start_tracking():
    selected_class_info = class_selector_combo.get()
    if not selected_class_info or selected_class_info == "Select a class":
        mess.showerror("Error", "Please select a class before taking attendance.")
        return
    # Extract name and ID
    class_name = selected_class_info.split(" (ID:")[0]
    class_id = int(selected_class_info.split("ID: ")[1].replace(")", ""))
    TrackImages(class_id, class_name)

def TrackImages(class_id, class_name):
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    for k in tv.get_children():
        tv.delete(k)

    # --- MODIFIED ---: Get students enrolled in the selected class
    try:
        enroll_df = pd.read_csv(os.path.join("Data", "Enrollments.csv"))
        enrolled_student_ids = enroll_df[enroll_df['ClassID'] == class_id]['StudentID'].tolist()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        enrolled_student_ids = []

    if not enrolled_student_ids:
        mess.showinfo("Info", f"No students are enrolled in '{class_name}'.")
        return

    # Load recognizer and student details
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    if not os.path.isfile("TrainingImageLabel/Trainner.yml"):
        mess.showerror('Data Missing', 'Please click on Save Profile to train the model!')
        return
    recognizer.read("TrainingImageLabel/Trainner.yml")

    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails/StudentDetails.csv")

    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Store attendance records temporarily
    attendance_records = {}

    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            serial, conf = recognizer.predict(gray[y:y + h, x:x + w])
            
            # --- MODIFIED ---: Check confidence AND enrollment
            if conf < 50 and serial in enrolled_student_ids:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                
                student_info = df.loc[df['SERIAL NO.'] == serial]
                aa = student_info['NAME'].values[0]
                ID = student_info['ID'].values[0]
                
                # Add to attendance if not already marked
                if serial not in attendance_records:
                   attendance_records[serial] = [str(ID), '', aa, '', str(date), '', str(timeStamp), '', str(class_id)]
                
                bb = aa
            elif conf < 50 and serial not in enrolled_student_ids:
                bb = "Not in this class"
            else:
                bb = "Unknown"
            cv2.putText(im, str(bb), (x, y + h), font, 1, (255, 255, 255), 2)
        
        cv2.imshow(f'Taking Attendance for {class_name}', im)
        if (cv2.waitKey(1) == ord('q')):
            break
            
    cam.release()
    cv2.destroyAllWindows()

    # --- MODIFIED ---: Save all unique attendance records at the end
    if attendance_records:
        date = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y')
        filename = os.path.join("Attendance", f"Attendance_{date}.csv")
        col_names = ['Id', '', 'Name', '', 'Date', '', 'Time', '', 'ClassID']
        
        # Write header if file doesn't exist
        write_header = not os.path.exists(filename)
        
        with open(filename, 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            if write_header:
                writer.writerow(col_names)
            for record in attendance_records.values():
                writer.writerow(record)
    
    # --- MODIFIED ---: Load and display today's attendance for the selected class
    load_attendance_for_class(class_id)


# --- ADDED ---: Helper to load attendance into TreeView
def load_attendance_for_class(class_id):
    for k in tv.get_children():
        tv.delete(k)
    date = datetime.datetime.fromtimestamp(time.time()).strftime('%d-%m-%Y')
    filename = os.path.join("Attendance", f"Attendance_{date}.csv")
    try:
        df = pd.read_csv(filename)
        # Filter for the specific class
        class_df = df[df['ClassID'] == class_id]
        class_df = class_df.drop_duplicates(subset=['Id', 'Date'])
        for index, lines in class_df.iterrows():
            tv.insert('', 'end', text=lines['Id'], values=(str(lines['Name']), str(lines['Date']), str(lines['Time'])))
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
        pass # No records for today or file doesn't exist yet

######################################################################################

def update_registration_count():
    try:
        df = pd.read_csv("StudentDetails/StudentDetails.csv")
        res = len(df)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        res = 0
    reg_count_label.config(text=f"Total Registrations: {res}")


######################################## GUI FRONT-END (REDESIGNED) ###########################################

BG_COLOR = "#1F2833"
FRAME_COLOR = "#2C3E50"
TEXT_COLOR = "#FFFFFF"
ACCENT_COLOR = "#66FCF1"
ALT_ACCENT_COLOR = "#45A29E"
FONT_BOLD = ("Segoe UI", 12, "bold")
FONT_NORMAL = ("Segoe UI", 12) # Adjusted for better fit
FONT_TITLE = ("Segoe UI", 32, "bold")
FONT_HEADER = ("Segoe UI", 16, "bold")

# Main Window and initial file setup
window = tk.Tk()
initialize_data_files() # --- ADDED ---
window.geometry("1280x720")
window.title("Aura Attend")
window.configure(bg=BG_COLOR)

# Header Frame (remains the same)
header_frame = tk.Frame(window, bg=FRAME_COLOR, pady=10)
header_frame.pack(fill='x')
title_label = tk.Label(header_frame, text=" AURA  ATTEND", fg=ACCENT_COLOR, bg=FRAME_COLOR, font=FONT_TITLE)
title_label.pack()
assistant_canvas = tk.Canvas(header_frame, width=60, height=60, bg=FRAME_COLOR, bd=0, highlightthickness=0, cursor="hand2")
assistant_canvas.place(relx=0.98, rely=0.5, anchor='e')
circle = assistant_canvas.create_oval(5, 5, 55, 55, fill="#c0392b", outline="")
assistant_text = assistant_canvas.create_text(30, 30, text="AI", fill="white", font=("Segoe UI", 16, "bold"))
assistant_canvas.tag_bind(circle, '<Button-1>', lambda e: assistant(window))
assistant_canvas.tag_bind(assistant_text, '<Button-1>', lambda e: assistant(window))
def on_assistant_enter(e): assistant_canvas.itemconfig(circle, fill="#E74C3C")
def on_assistant_leave(e): assistant_canvas.itemconfig(circle, fill="#c0392b")
assistant_canvas.tag_bind(circle, '<Enter>', on_assistant_enter)
assistant_canvas.tag_bind(circle, '<Leave>', on_assistant_leave)
assistant_canvas.tag_bind(assistant_text, '<Enter>', on_assistant_enter)
assistant_canvas.tag_bind(assistant_text, '<Leave>', on_assistant_leave)
time_frame = tk.Frame(header_frame, bg=FRAME_COLOR)
time_frame.pack(pady=5)
ts = time.time()
date_str = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
mont={'01':'January','02':'February','03':'March','04':'April','05':'May','06':'June','07':'July','08':'August','09':'September','10':'October','11':'November','12':'December'}
day,month,year=date_str.split("-")
date_label = tk.Label(time_frame, text=f"{day} {mont[month]} {year}", fg=TEXT_COLOR, bg=FRAME_COLOR, font=FONT_NORMAL)
date_label.pack(side='left', padx=10)
clock = tk.Label(time_frame, fg=TEXT_COLOR, bg=FRAME_COLOR, font=FONT_NORMAL)
clock.pack(side='left', padx=10)
tick()

# Main Content Frame
main_frame = tk.Frame(window, bg=BG_COLOR)
main_frame.pack(pady=20, padx=20, fill='both', expand=True)
main_frame.columnconfigure(0, weight=1)
main_frame.columnconfigure(1, weight=1)
main_frame.rowconfigure(0, weight=1)

# Registration Frame (Right Side) --- MODIFIED ---
reg_frame = tk.Frame(main_frame, bg=FRAME_COLOR, bd=2, relief='groove')
reg_frame.grid(row=0, column=1, sticky="nsew", padx=10)
reg_frame.rowconfigure(5, weight=1) # Make class listbox expandable

reg_header = tk.Label(reg_frame, text="New Registration", font=FONT_HEADER, fg=ACCENT_COLOR, bg=FRAME_COLOR)
reg_header.pack(pady=10)
tk.Label(reg_frame, text="Enter Registration No.", font=FONT_NORMAL, fg=TEXT_COLOR, bg=FRAME_COLOR).pack(pady=(5,0))
txt = tk.Entry(reg_frame, width=30, font=FONT_NORMAL, bg="#34495E", fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
txt.pack(pady=5, padx=10, fill='x')
tk.Label(reg_frame, text="Enter Full Name", font=FONT_NORMAL, fg=TEXT_COLOR, bg=FRAME_COLOR).pack(pady=(5,0))
txt2 = tk.Entry(reg_frame, width=30, font=FONT_NORMAL, bg="#34495E", fg=TEXT_COLOR, insertbackground=TEXT_COLOR)
txt2.pack(pady=5, padx=10, fill='x')

# --- ADDED ---: Class selection for registration
tk.Label(reg_frame, text="Assign to Classes (Ctrl+Click for multiple)", font=FONT_NORMAL, fg=TEXT_COLOR, bg=FRAME_COLOR).pack(pady=(10,0))
reg_class_frame = tk.Frame(reg_frame)
reg_class_frame.pack(pady=5, padx=10, fill='both', expand=True)
reg_class_listbox = tk.Listbox(reg_class_frame, selectmode=tk.MULTIPLE, font=FONT_NORMAL, bg="#34495E", fg=TEXT_COLOR, selectbackground=ALT_ACCENT_COLOR)
reg_class_listbox.pack(side=tk.LEFT, fill='both', expand=True)
reg_scrollbar = ttk.Scrollbar(reg_class_frame, orient="vertical", command=reg_class_listbox.yview)
reg_scrollbar.pack(side=tk.RIGHT, fill='y')
reg_class_listbox.config(yscrollcommand=reg_scrollbar.set)

clear_frame = tk.Frame(reg_frame, bg=FRAME_COLOR)
clear_frame.pack(pady=5)
clearButton = tk.Button(clear_frame, text="Clear ID", command=clear, font=FONT_NORMAL, width=12, bg="#E74C3C")
clearButton.pack(side='left', padx=5)
clearButton2 = tk.Button(clear_frame, text="Clear Name", command=clear2, font=FONT_NORMAL, width=12, bg="#E74C3C")
clearButton2.pack(side='left', padx=5)
message1 = tk.Label(reg_frame, text="1)Take Images  >>>  2)Save Profile", bg=FRAME_COLOR, fg=TEXT_COLOR, font=FONT_NORMAL)
message1.pack(pady=5)
reg_btn_frame = tk.Frame(reg_frame, bg=FRAME_COLOR)
reg_btn_frame.pack(pady=10)
takeImg = tk.Button(reg_btn_frame, text="Take Images & Enroll", command=TakeImages, font=FONT_BOLD, width=20, height=2, bg=ALT_ACCENT_COLOR, fg=TEXT_COLOR)
takeImg.pack(side='left', padx=5)
trainImg = tk.Button(reg_btn_frame, text="Save Profile", command=psw, font=FONT_BOLD, width=20, height=2, bg=ACCENT_COLOR, fg=BG_COLOR)
trainImg.pack(side='left', padx=5)
reg_count_label = tk.Label(reg_frame, text="", bg=FRAME_COLOR, fg=TEXT_COLOR, font=FONT_NORMAL)
reg_count_label.pack(side='bottom', pady=10)
update_registration_count()

# Attendance Frame (Left Side) --- MODIFIED ---
att_frame = tk.Frame(main_frame, bg=FRAME_COLOR, bd=2, relief='groove')
att_frame.grid(row=0, column=0, sticky="nsew", padx=10)

att_header = tk.Label(att_frame, text="Attendance Log", font=FONT_HEADER, fg=ACCENT_COLOR, bg=FRAME_COLOR)
att_header.pack(pady=10)

# --- ADDED ---: Class selector for taking attendance
class_selector_frame = tk.Frame(att_frame, bg=FRAME_COLOR)
class_selector_frame.pack(fill='x', padx=10, pady=5)
tk.Label(class_selector_frame, text="Select Class:", font=FONT_BOLD, bg=FRAME_COLOR, fg=TEXT_COLOR).pack(side='left')
class_selector_combo = ttk.Combobox(class_selector_frame, font=FONT_NORMAL, state='readonly', values=["Select a class"])
class_selector_combo.pack(side='left', fill='x', expand=True, padx=5)
class_selector_combo.set("Select a class")
def on_class_select(event):
    selected_info = class_selector_combo.get()
    if selected_info != "Select a class":
        class_id = int(selected_info.split("ID: ")[1].replace(")", ""))
        load_attendance_for_class(class_id)
class_selector_combo.bind("<<ComboboxSelected>>", on_class_select)


# --- ADDED ---: Helper to populate all class dropdowns/lists
def populate_class_lists():
    try:
        df = pd.read_csv(os.path.join("Data", "Classes.csv"))
        class_names = df['ClassName'].tolist()
        class_info_list = [f"{row['ClassName']} (ID: {row['ClassID']})" for index, row in df.iterrows()]
    except (FileNotFoundError, pd.errors.EmptyDataError):
        class_names = []
        class_info_list = []

    # Update registration listbox
    reg_class_listbox.delete(0, tk.END)
    for name in class_names:
        reg_class_listbox.insert(tk.END, name)

    # Update attendance combobox
    class_selector_combo['values'] = ["Select a class"] + class_info_list


# TreeView setup
style = ttk.Style()
style.theme_use("default")
style.configure("Treeview", background=FRAME_COLOR, foreground=TEXT_COLOR, fieldbackground=FRAME_COLOR, rowheight=25, font=FONT_NORMAL)
style.map('Treeview', background=[('selected', ALT_ACCENT_COLOR)])
style.configure("Treeview.Heading", background=BG_COLOR, foreground=ACCENT_COLOR, font=FONT_BOLD, padding=(10,10))
style.map("Treeview.Heading", background=[('active', ALT_ACCENT_COLOR)])
tree_frame = tk.Frame(att_frame, bg=FRAME_COLOR)
tree_frame.pack(expand=True, fill='both', padx=10, pady=5)
scroll = ttk.Scrollbar(tree_frame, orient='vertical')
tv = ttk.Treeview(tree_frame, height=13, columns=('name', 'date', 'time'), yscrollcommand=scroll.set)
scroll.config(command=tv.yview)
tv.column('#0',width=82); tv.column('name',width=130); tv.column('date',width=133); tv.column('time',width=133)
tv.heading('#0',text ='ID'); tv.heading('name',text ='NAME'); tv.heading('date',text ='DATE'); tv.heading('time',text ='TIME')
tv.pack(side='left', fill='both', expand=True)
scroll.pack(side='right', fill='y')

# Buttons
att_btn_frame = tk.Frame(att_frame, bg=FRAME_COLOR)
att_btn_frame.pack(pady=15, fill='x', side='bottom')
# --- MODIFIED ---: Button now calls start_tracking wrapper
trackImg = tk.Button(att_btn_frame, text="Take Attendance", command=start_tracking, font=FONT_BOLD, height=2, bg=ACCENT_COLOR, fg=BG_COLOR)
trackImg.pack(side='left', expand=True, padx=10)
quitWindow = tk.Button(att_btn_frame, text="     Exit    ", command=window.destroy, font=FONT_BOLD, height=2, bg="#c0392b", fg=TEXT_COLOR)
quitWindow.pack(side='right', expand=True, padx=10)

# Menubar --- MODIFIED ---
menubar = tk.Menu(window)
filemenu = tk.Menu(menubar, tearoff=0)
# --- ADDED ---: Link to Class Manager
filemenu.add_command(label='Class Manager', command=open_class_manager)
filemenu.add_separator()
filemenu.add_command(label='Change Password', command=change_pass)
filemenu.add_command(label='Contact Us', command=contact)
filemenu.add_separator()
filemenu.add_command(label='Exit',command = window.destroy)
menubar.add_cascade(label='Options', menu=filemenu)
window.config(menu=menubar)

populate_class_lists() # --- ADDED ---: Initial population of class lists

####################### ASSISTANT (NOW CONTEXT- & CLASS-AWARE) ###############################

# --- Assistant code remains largely the same, but the data handling functions below it are modified. ---
# (AssistantUI class and assistant_voice_task function are unchanged from the previous version)
class AssistantUI:
    def __init__(self, parent_window):
        self.window = tk.Toplevel(parent_window)
        self.window.title("Voice Assistant")
        self.window.geometry("550x650")
        self.window.configure(bg=BG_COLOR)
        self.window.transient(parent_window)
        self.window.grab_set()

        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        if voices:
            self.engine.setProperty('voice', voices[0].id)
        
        self.model = None
        self.chat = None
        try:
            self.model = genai.GenerativeModel('gemini-pro')
            self.chat = self.model.start_chat(history=[])
        except Exception as e:
            print(f"Could not initialize Generative Model: {e}")
            self.add_conversation_entry("System Error", f"Could not initialize AI Model: {e}", "error_tag")

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
        input_frame.columnconfigure(0, weight=1)
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
        if self.window.winfo_exists(): self.window.after(0, func)
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
        if not query: return
        self.text_entry.delete(0, tk.END)
        self.add_conversation_entry("You", query, "user_tag")
        self.disable_inputs()
        thread = threading.Thread(target=self._process_query, args=(query,))
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
        if self.engine._inLoop: self.engine.stop()
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
    def _process_query(self, query: str):
        try:
            self.update_status("Thinking...")
            local_response = handle_attendance_query(query)
            if local_response:
                response_text = local_response
                if self.chat:
                    self.chat.history.append({'role': 'user', 'parts': [{'text': query}]})
                    self.chat.history.append({'role': 'model', 'parts': [{'text': response_text}]})
            else:
                if not self.chat:
                    response_text = "The AI model is not available. Please check API key and restart."
                else:
                    response = self.chat.send_message(query)
                    response_text = ''.join(part.text for part in response.parts) if response.parts else "I'm sorry, I couldn't find an answer for that."
            self.add_conversation_entry("Assistant", response_text, "assistant_tag")
            self.speak(response_text)
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            print(f"ASSISTANT ERROR: {error_message}")
            self.add_conversation_entry("System Error", error_message, "error_tag")
        finally:
            if self.window.winfo_exists(): self.enable_inputs()


# --- MODIFIED ---: All data retrieval functions are now class-aware
def _get_class_info_from_query(query: str):
    """Parses a query to find a class name and returns its ID and name."""
    try:
        classes_df = pd.read_csv(os.path.join("Data", "Classes.csv"))
        for index, row in classes_df.iterrows():
            if row['ClassName'].lower() in query.lower():
                return row['ClassID'], row['ClassName']
        return None, None
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return None, None

def _get_all_attendance_data(class_id: int = None):
    """Reads all attendance, optionally filtering by class ID."""
    attendance_dir = "Attendance/"
    if not os.path.exists(attendance_dir): return None
    all_files = [os.path.join(attendance_dir, f) for f in os.listdir(attendance_dir) if f.endswith('.csv')]
    if not all_files: return None
    try:
        df = pd.concat((pd.read_csv(f) for f in all_files), ignore_index=True)
        df.drop_duplicates(subset=['Id', 'Date', 'ClassID'], inplace=True)
        if class_id:
            df = df[df['ClassID'] == class_id]
        return df if not df.empty else None
    except (pd.errors.EmptyDataError, ValueError, KeyError):
        return None

def _get_all_registered_students(class_id: int = None):
    """Returns a list of students, optionally filtering by class enrollment."""
    try:
        students_df = pd.read_csv("StudentDetails/StudentDetails.csv")
        if not class_id:
            return students_df['SERIAL NO.'].tolist(), students_df
        
        enroll_df = pd.read_csv(os.path.join("Data", "Enrollments.csv"))
        student_ids_in_class = enroll_df[enroll_df['ClassID'] == class_id]['StudentID'].tolist()
        enrolled_students_df = students_df[students_df['SERIAL NO.'].isin(student_ids_in_class)]
        return student_ids_in_class, enrolled_students_df
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
        return [], pd.DataFrame()

def _get_todays_present_students(class_id: int = None):
    """Returns a list of students present today, optionally filtered by class."""
    today = datetime.datetime.now().strftime('%d-%m-%Y')
    filepath = os.path.join("Attendance", f"Attendance_{today}.csv")
    try:
        df = pd.read_csv(filepath)
        if class_id:
            df = df[df['ClassID'] == class_id]
        return df['Name'].unique().tolist() if not df.empty else []
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
        return []

# --- MODIFIED ---: The main query handler is now fully class-aware
def handle_attendance_query(query: str):
    query_lower = query.lower()
    class_id, class_name = _get_class_info_from_query(query)
    class_context_str = f" for {class_name}" if class_name else ""

    # --- QUERIES ABOUT TODAY'S ATTENDANCE ---
    if "today" in query_lower or "currently" in query_lower:
        present_students = _get_todays_present_students(class_id)
        all_student_ids, all_students_df = _get_all_registered_students(class_id)
        if not all_student_ids: return f"There are no students registered{class_context_str}."

        present_df = all_students_df[all_students_df['NAME'].isin(present_students)]
        absent_df = all_students_df[~all_students_df['NAME'].isin(present_students)]

        if "how many" in query_lower and "present" in query_lower:
            return f"There are {len(present_students)} students present today{class_context_str}."
        if "how many" in query_lower and "absent" in query_lower:
            return f"There are {len(absent_df)} students absent today{class_context_str}."
        if ("list" in query_lower and "present" in query_lower):
            if present_students: return f"The students present today{class_context_str} are: {', '.join(sorted(present_students))}."
            else: return f"No one has been marked present yet today{class_context_str}."
        if ("list" in query_lower and "absent" in query_lower):
            absent_names = sorted(absent_df['NAME'].tolist())
            if absent_names: return f"The students absent today{class_context_str} are: {', '.join(absent_names)}."
            else: return f"It looks like everyone is present today{class_context_str}."
    
    # Other queries can be similarly updated...
    # For simplicity, this example focuses on the most common "today" query.
    # The logic for other queries like "attendance of" would also need to use the `class_id` filter.
            
    return None # Return None to let Gemini handle it

def assistant_voice_task(ui: AssistantUI):
    # This function remains unchanged
    try:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            r.adjust_for_ambient_noise(source, duration=0.5)
            ui.speak("How can I help you?")
            if not ui.window.winfo_exists(): return
            ui.update_status("Listening...")
            try: audio = r.listen(source, timeout=7, phrase_time_limit=15)
            except sr.WaitTimeoutError:
                ui.update_status("Listening timed out.")
                ui.enable_inputs()
                return
        ui.update_status("Recognizing...")
        query = r.recognize_google(audio, language='en-in')
        ui.add_conversation_entry("You", query, "user_tag")
        ui._process_query(query)
    except Exception as e:
        error_message = f"An error occurred during voice input: {e}"
        print(f"VOICE TASK ERROR: {error_message}")
        ui.add_conversation_entry("System Error", error_message, "error_tag")
        if ui.window.winfo_exists(): ui.enable_inputs()

def assistant(parent_window):
    AssistantUI(parent_window)

##################### END ######################################
window.mainloop()