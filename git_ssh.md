cd .ssh folder

ssh-keygen -t rsa -C "jorgelopez301" -f "jorgelopez301"

add ssh key to github setting

eval "$(ssh-agent -s)"

ssh-add ~/.ssh/jorgelopez301

make config file
//C:\Users\Administrator\.ssh\config

#chris-account

```
Host github.com
    HostName github.com
    User git
    IdentityFile ~/.ssh/jorgelopez301
```
