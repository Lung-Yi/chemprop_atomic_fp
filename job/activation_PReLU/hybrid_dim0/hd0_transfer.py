with open("hd0_template_transfer.sh", 'r') as g:
    text = g.read()
print(text)
for n in range(1,10):
    new_text = text.replace("$$$NUM", str(n))
    with open("hd0_PReLU_{}_transfer.sh".format(str(n)), 'w') as f:
        f.write(new_text)

