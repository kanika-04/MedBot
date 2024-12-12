import streamlit as st
import json
import os
from serial_connection import send_coordinates

# File paths for user credentials and medicine coordinates
DOCTOR_CREDENTIALS_FILE = "doctors.json"
PATIENT_CREDENTIALS_FILE = "patients.json"
MEDICINE_COORDINATE_FILE = "medicine_coordinates.json"

def load_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    else:
        return {}

def save_data(file_path, data):
    with open(file_path, "w") as file:
        json.dump(data, file)

def login(username, password, credentials):
    return username in credentials and credentials[username]['password'] == password

def register(username, password, credentials_file, user_type):
    credentials = load_data(credentials_file)
    if username in credentials:
        st.warning(f"{user_type} username already exists! Please choose a different one.")
    else:
        credentials[username] = {"password": password}
        if user_type == "Doctor":
            credentials[username]["medicines"] = {}
        save_data(credentials_file, credentials)
        st.success(f"{user_type} registration successful!")

def assign_medicine(patient_id, selected_medicines, doctor_credentials, doctor_username):
    if patient_id not in doctor_credentials[doctor_username]["medicines"]:
        doctor_credentials[doctor_username]["medicines"][patient_id] = []

    for med in selected_medicines:
        doctor_credentials[doctor_username]["medicines"][patient_id].append({
            "medicine": med,
            "prescribed_by": doctor_username
        })
    
    save_data(DOCTOR_CREDENTIALS_FILE, doctor_credentials)
    st.success(f"Assigned medicines {selected_medicines} to patient ID {patient_id}.")

def collect_medicine(patient_id, doctor_credentials, medicine_coordinates):
    # Initialize the x and y coordinates lists and the found_medicines flag
    x_values = []
    y_values = []
    found_medicines = False

    # Retrieve and collect the coordinates for the assigned medicines
    for doctor in doctor_credentials:
        if patient_id in doctor_credentials[doctor]["medicines"]:
            found_medicines = True
            for entry in doctor_credentials[doctor]["medicines"][patient_id]:
                med = entry["medicine"]
                coordinate = medicine_coordinates.get(med, "Unknown Coordinates")
                if coordinate != "Unknown Coordinates":
                    x, y = map(int, coordinate.split(","))
                    x_values.append(x)
                    y_values.append(y)

    # Send coordinates only if medicines were found and not already collected
    if found_medicines:
        if "collect_clicked" not in st.session_state or not st.session_state["collect_clicked"]:
            st.success("Medicines collected. Sending coordinates through serial connection.")
            send_coordinates(x_values, y_values)
            st.session_state["collect_clicked"] = True  # Mark the button as clicked
        else:
            st.warning("Medicines have already been collected.")
    else:
        st.warning("No medicines assigned yet.")






def view_patient_medicines(patient_id, doctor_credentials):
    st.write(f"Medicines assigned to patient ID {patient_id}:")
    found_medicines = False
    for doctor in doctor_credentials:
        if patient_id in doctor_credentials[doctor]["medicines"]:
            found_medicines = True
            for entry in doctor_credentials[doctor]["medicines"][patient_id]:
                st.write(f"{entry['medicine']} - Prescribed by: {entry['prescribed_by']}")
    if not found_medicines:
        st.write("No medicines assigned yet.")

def doctor_dashboard(doctor_username):
    st.title("Doctor Dashboard")
    st.write("Welcome, Doctor! Here you can assign medicines to patients.")
    if st.button("Go to Assign Medicines"):
        st.session_state["page"] = "assign_medicines"
        st.rerun()
    
    if st.button("Logout"):
        logout()

def assign_medicines_page(doctor_credentials, medicine_coordinates, doctor_username):
    st.title("Assign Medicines to Patients")

    patient_id = st.text_input("Enter Patient ID")
    medicines = list(medicine_coordinates.keys())
    selected_medicines = st.multiselect("Choose medicines to assign", medicines)

    if st.button("Assign Medicines"):
        assign_medicine(patient_id, selected_medicines, doctor_credentials, doctor_username)

    if st.button("Back to Dashboard"):
        st.session_state["page"] = "doctor_dashboard"
        st.rerun()

def patient_dashboard(patient_id, doctor_credentials, medicine_coordinates):
    st.title("Patient Medicine Tracker")

    if st.button("Check Medicines"):
        view_patient_medicines(patient_id, doctor_credentials)

    if st.button("Collect Medicines"):
        collect_medicine(patient_id, doctor_credentials, medicine_coordinates)

    if st.button("Logout"):
        logout()

def logout():
    st.session_state.clear()
    st.session_state["page"] = "home"
    st.rerun()

# Load data files
doctor_credentials = load_data(DOCTOR_CREDENTIALS_FILE)
patient_credentials = load_data(PATIENT_CREDENTIALS_FILE)
medicine_coordinates = load_data(MEDICINE_COORDINATE_FILE)

# Page navigation logic
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# Home Page
if st.session_state["page"] == "home":
    st.title("Medicine Tracker App")
    user_type = st.radio("Choose your role", ("Doctor", "Patient"))

    if user_type == "Doctor":
        st.subheader("Doctor Login")
        doctor_username = st.text_input("Username")
        doctor_password = st.text_input("Password", type="password")
        if st.button("Login as Doctor"):
            if login(doctor_username, doctor_password, doctor_credentials):
                st.session_state["logged_in"] = True
                st.session_state["user_type"] = "Doctor"
                st.session_state["username"] = doctor_username
                st.session_state["page"] = "doctor_dashboard"
                st.rerun()
            else:
                st.error("Invalid username or password")
        elif st.button("Register as Doctor"):
            register(doctor_username, doctor_password, DOCTOR_CREDENTIALS_FILE, "Doctor")

    elif user_type == "Patient":
        st.subheader("Patient Login")
        patient_id = st.text_input("Patient ID")
        patient_password = st.text_input("Password", type="password")
        if st.button("Login as Patient"):
            if login(patient_id, patient_password, patient_credentials):
                st.session_state["logged_in"] = True
                st.session_state["user_type"] = "Patient"
                st.session_state["patient_id"] = patient_id
                st.session_state["page"] = "patient_dashboard"
                st.rerun()
            else:
                st.error("Invalid patient ID or password")
        elif st.button("Register as Patient"):
            register(patient_id, patient_password, PATIENT_CREDENTIALS_FILE, "Patient")

# Doctor Dashboard
elif st.session_state["page"] == "doctor_dashboard":
    doctor_dashboard(st.session_state["username"])

# Assign Medicines Page
elif st.session_state["page"] == "assign_medicines":
    assign_medicines_page(doctor_credentials, medicine_coordinates, st.session_state["username"])

# Patient Dashboard
elif st.session_state["page"] == "patient_dashboard":
    patient_dashboard(st.session_state["patient_id"], doctor_credentials, medicine_coordinates)
