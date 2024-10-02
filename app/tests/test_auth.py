import pytest
from app import create_app

@pytest.fixture
def client():
    app = create_app()
    app.config['TESTING'] = True
    app.config['MYSQL_DB'] = 'flaskdb'  # Set your test database if needed

    with app.test_client() as client:
        yield client

# def test_login_page(client):
#     """Test the login page"""
#     response = client.get('/auth/login')
#     assert response.status_code == 200
#     assert b'Login' in response.data  # Check if 'Login' is in the response

# def test_register_page(client):
#     """Test the registration page"""
#     response = client.get('/auth/register')
#     assert response.status_code == 200
#     assert b'Register' in response.data

def test_successful_registration(client):
    """Test successful user registration"""
    response = client.post('/auth/register', data={
        'username': 'testuser',
        'email': 'test@example.com',
        'password': 'testpassword'
    })
    print(response.data)
    assert response.status_code == 302  # Check for redirection after successful registration

# def test_successful_login(client):
#     """Test successful user login"""
#     # First, register a user
#     client.post('/auth/register', data={
#         'username': 'testuser',
#         'email': 'test@example.com',
#         'password': 'testpassword'
#     })

#     # Now, log in
#     response = client.post('/auth/login', data={
#         'username': 'testuser',
#         'password': 'testpassword'
#     })
#     assert response.status_code == 302  # Check for redirection after successful login
