from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    platform_name = models.CharField(max_length=50)
    access_token = models.TextField()

class Message(models.Model):
    user_profile = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    message_text = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    response_text = models.TextField(blank=True, null=True)
