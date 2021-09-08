+++
draft = true
title = "Concurrency in Python with FastAPI"
date = 2021-09-10

[taxonomies]
categories = ["concurrency"]
tags = ["python", "programming", "concurrency", "parallelism"]
+++

I've struggled for a long time with concurrency and parallelism. Let's dive in with the hot-cool-new [ASGI](https://asgi.readthedocs.io/en/latest/) framework, [FastAPI](https://fastapi.tiangolo.com/).

First things first, let's install FastAPI by following the [guide](https://fastapi.tiangolo.com/#installation). Once this is done, create the following `server.py` file:

```python
# server.py

import random
import time

from fastapi import FastAPI

app = FastAPI()


@app.get("/wait")
def wait():
    duration = 1.
    time.sleep(duration)
    return {"duration": duration}
```

Run it with

```bash
uvicorn server:app --reload
```

You should see at `http://127.0.0.1:8000/wait` something like:

```json
{ "duration": 1 }
```

If we want to simply assess the overhead of the HTTP protocol, we can use the apache benchmarking (`ab`) tool with this route:

```bash
ab -n 10 127.0.0.1:8000/wait
```

Which gives us a mean overhead of 2.5 ms for 10 sequential requests. Not bad!

Now let's try with _concurrent_ requests:

```bash
ab -n 10 -c 10 127.0.0.1:8000/wait
```
