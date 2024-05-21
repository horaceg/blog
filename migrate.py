from pathlib import Path

posts = Path("content/posts/")

for post in posts.rglob("*.md"):
    relto = post.relative_to("content")
    print(relto)
    with open(post, "r") as f:
        content = f.read()
    with open(Path("bl/") / relto, "w") as f:
        f.write(content.replace("+++", "---"))
