import imaplib
import email
import os
import time

MAIL_SERVER = 'imap.gmail.com'
EMAIL_USER = 'shrutj2104@gmail.com'
EMAIL_PASS = 'aymw wxrh fmea bxfz'
MAIL_BOX = 'INBOX'
DOWNLOAD_FOLDER = "./downloaded_pdfs"

os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

def download_attached_file(mail,mail_box,download_folder):
    mail.select(mail_box)

    status, mails = mail.search(None,'UNSEEN')
    email_ids = mails[0].split()

    if not email_ids:
        print("No new emails found.")
        return
    
    saved_files = []
    print(f"Found {len(email_ids)} new emails.")
    for eid in email_ids:
        _, msg_data = mail.fetch(eid, "(RFC822)")
        msg = email.message_from_bytes(msg_data[0][1])

        for part in msg.walk():
            # multipart are containers, bypass them
            if part.get_content_maintype() == 'multipart':
                continue
            # Only process parts with attachments
            if part.get('Content-Disposition') is None:
                continue

            filename = part.get_filename()
            if filename and filename.lower().endswith(".pdf"):
                filepath = os.path.join(download_folder, filename)
                # Avoid overwriting by adding suffix if file exists
                
                if not os.path.exists(filepath):
                    with open(filepath, 'wb') as f:
                        f.write(part.get_payload(decode=True))
                    print(f"Saved attachment: {filepath}")
                    saved_files.append(filepath)

    
    return saved_files

if __name__ == "__main__":
    mail = imaplib.IMAP4_SSL(MAIL_SERVER)
    mail.login(EMAIL_USER, EMAIL_PASS)
    print("Logged in successfully.")

    while True:
        downloaded_files = download_attached_file(mail,MAIL_BOX, DOWNLOAD_FOLDER)
        if downloaded_files:
            print("Downloaded files:", downloaded_files)
        else:
            print("No PDF attachments found.")

        time.sleep(10)  # Check for new emails every 60 seconds
    