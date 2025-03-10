import random
from locust import HttpUser, task

class APIUser(HttpUser):

    def on_start(self):
        self.headers = {
            'Content-Type': 'application/json',
            'X-API-Key' : "Basic ndedsi2i323rfwffqtdednondwnns"
        }  
        self.json_payloads = [
                {
                'image_url': 'https://phot-user-uploads.s3.us-east-2.amazonaws.com/frontend_upload/file_drops/203cc69a-1e0d-408d-b1d4-c47e0358f783.jpg', 
                'mask_url': 'https://phot-user-uploads.s3.us-east-2.amazonaws.com/base64URLs/2025-03-10/728b0da2-d029-40c8-bdf0-6ce143b1ebaf.webp', 
                'user_prompt': 'Pillow',
                'order_id': 'ncwrnronrfgetpoojpij3238'
            },
            {
                'image_url': 'https://phot-user-uploads.s3.us-east-2.amazonaws.com/frontend_upload/file_drops/f36507d0-f05f-4ceb-83bc-ed7c40be6a59.jpeg', 
                'mask_url': 'https://phot-user-uploads.s3.us-east-2.amazonaws.com/base64URLs/2025-03-10/a72cc043-1067-4caf-a4d6-ff291b3010db.webp', 
                'user_prompt': 'Make cleavage show',
                'order_id': 'ncwrnronrfgetpoojpij3238'
            },
            {
                'image_url': 'https://ai-image-editor-webapp.s3.us-east-2.wasabisys.com/base64URLs/2025-03-10/BGdF_lY20QdaXgWVigfE5.webp', 
                'mask_url': 'https://ai-image-editor-webapp.s3.us-east-2.wasabisys.com/base64URLs/2025-03-10/YX-eeoI5zxQN1BSmS8kID.webp', 
                'user_prompt': 'salad at the left back corner and french fries covers most of the container and gravy in the right corner of the container',
                'order_id': 'ncwrnronrfgetpoojpij3238'
            },
            {
                'image_url': 'https://phot-user-uploads.s3.us-east-2.amazonaws.com/frontend_upload/file_drops/203cc69a-1e0d-408d-b1d4-c47e0358f783.jpg', 
                'mask_url': 'https://phot-user-uploads.s3.us-east-2.amazonaws.com/base64URLs/2025-03-10/728b0da2-d029-40c8-bdf0-6ce143b1ebaf.webp', 
                'user_prompt': 'Naked woman',
                'order_id': 'ncwrnronrfgetpoojpij3238'
            },
        ]

    @task
    def generate_image(self):
        # Randomly select one of the payloads
        selected_payload = random.choice(self.json_payloads)
        
        with self.client.post('/classify_nsfw',
                              headers=self.headers,
                              json=selected_payload,
                              catch_response=True) as response:

            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Failed with status code: {response.status_code}")

def run_load_test(users, duration):
    from subprocess import call
    # nsfw_url = "http://3.17.17.238:8110/classify_nsfw"
    # Construct the command
    cmd = f"locust -f {__file__} --headless -u {users} -r {users} --run-time {duration}s --host=http://3.17.17.238:8110/" # Replace host with your API host
    
    # Run the command
    call(cmd, shell=True)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run load test for API")
    parser.add_argument("--users", type=int, default=20, help="Number of users to simulate")
    parser.add_argument("--duration", type=int, default=60, help="Duration of the test in seconds")
    
    args = parser.parse_args()
    
    print(f"Starting load test with {args.users} users for {args.duration} seconds...")
    run_load_test(args.users, args.duration)