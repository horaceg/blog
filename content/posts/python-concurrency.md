+++
draft = true
title = "Concurrency in Python with FastAPI"
date = 2021-09-10

[taxonomies]
categories = ["concurrency"]
tags = ["python", "programming", "concurrency", "parallelism"]
+++

I've struggled for a long time with concurrency and parallelism.

```python
import asyncio
import functools
import random
import time
from concurrent.futures import ThreadPoolExecutor as Pool

import aiohttp
import requests


def timed(N, url, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = fn(*args, **kwargs)
        stop = time.time()
        duration = stop - start
        print(f"{N / duration:.1f} reqs / sec | {N} reqs | {url} | {fn.__name__}")
        return res

    return wrapper
```

Now everything is clear.
