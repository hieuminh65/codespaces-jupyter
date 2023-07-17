import base64

def convert_m4a_to_base64(filename):
    with open(filename, 'rb') as audio_file:
        # Read the entire file
        audio_data = audio_file.read()
        # Encode the data into base64
        base64_data = base64.b64encode(audio_data)
        # Convert to string
        base64_string = base64_data.decode('utf-8')
        return base64_string

# Example usage
base64_str = convert_m4a_to_base64('test1.m4a')

# Export to txt file
with open('string.txt', 'w') as f:
    f.write(base64_str)
