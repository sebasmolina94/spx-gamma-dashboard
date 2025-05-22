import schwabdev #import the package
client = schwabdev.Client('GnhaaTIat2rZn5RTdzOaDHeVNUs72fSf','kyirQ7e6mmO6wXkE') #create a client
print(client.account_linked().json())
