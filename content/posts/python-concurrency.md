+++
title = "Concurrency in Python with FastAPI"
date = 2021-09-10

[taxonomies]
categories = ["concurrency"]
tags = ["python", "programming", "concurrency", "parallelism"]
+++

I've struggled for a long time with concurrency and parallelism. Let's dive in with the hot-cool-new [ASGI](https://asgi.readthedocs.io/en/latest/) framework, [FastAPI](https://fastapi.tiangolo.com/). It is a concurrent framework, which means `asyncio`-friendly. Tiangolo, the author, claims that the performance is on par with Go and Node webservers. We're going to see a glimpse of the reason (spoilers: concurrency).

First things first, let's install FastAPI by following the [guide](https://fastapi.tiangolo.com/#installation). 

# Purely IO-bound workloads

We are going to simulate a pure IO operation, such as an waiting for a database to finish its operation. Let's create the following `server.py` file:

```python
# server.py

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

Ok, it works. Now, let's dive into the performance comparison. We could use [ApacheBench](https://en.wikipedia.org/wiki/ApacheBench), but here we are going to implement everything in python for the sake of clarity.

Let's create a `client.py` file:

```python
# client.py

import functools
import time

import requests


def timed(N, url, fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = fn(*args, **kwargs)
        stop = time.time()
        duration = stop - start
        print(f"{N / duration:.2f} reqs / sec | {N} reqs | {url} | {fn.__name__}")
        return res

    return wrapper


def get(url):
    resp = requests.get(url)
    assert resp.status_code == 200
    return resp.json()


def sync_get_all(url, n):
    l = [get(url) for _ in range(n)]
    return l


def run_bench(n, funcs, urls):
    for url in urls:
        for func in funcs:
            timed(n, url, func)(url, n)


if __name__ == "__main__":
    urls = ["http://127.0.0.1:8000/wait"]
    funcs = [sync_get_all]
    run_bench(10, funcs, urls)
```

Let's run this:

```bash
python client.py
```
```
0.99 reqs / sec | 10 reqs | http://127.0.0.1:8000/wait | sync_get_all
```

So far, we know that the overhead is sub-10 ms for ten requests, so less than 1ms per request. Cool!

## Threadpools client-side

Now, we are going to simulate multiple simultaneous connections. This is usually a problem we want to have: the more users of our web API or app, the more simultaneous requests. The previous test wasn't very realistic: users rarely browse sequentially, but rather appear simultaneously, forming bursts of activity.

We are going to implement concurrent requests using a __threadpool__:

```python
# client.py

...
from concurrent.futures import ThreadPoolExecutor as Pool
...


def thread_pool(url, n, limit=None):
    limit_ = limit or n
    with Pool(max_workers=limit_) as pool:
        result = pool.map(get, [url] * n)
    return result


if __name__ == "__main__":
    urls = ["http://127.0.0.1:8000/wait"]
    run_bench(10, [sync_get_all, thread_pool], urls)
```
We get:

```
0.99 reqs / sec | 10 reqs | http://127.0.0.1:8000/wait | sync_get_all
9.56 reqs / sec | 10 reqs | http://127.0.0.1:8000/wait | thread_pool
```

This looks 10x better! The overhead is of 44 ms for 10 requests, where does that come from?

Also, how come the server was able to answer asynchronously, since we only wrote synchronous (regular) Python code? There are no `async` nor `await`...

Well, this is how FastAPI works behind the scenes: it runs every synchronous request in a threadpool. So, we have threadpools both client-side and __server-side__!

Let's lower the duration:

```python, hl_lines=7
# server.py

...

@app.get("/wait")
def wait():
    duration = 0.05
    time.sleep(duration)
    return {"duration": duration}
```

Let's also run the benchmark 100 times:

```python, hl_lines=7
# client.py

...

if __name__ == "__main__":
    urls = ["http://127.0.0.1:8000/wait"]
    run_bench(100, [sync_get_all, thread_pool], urls)
```

```
15.91 reqs / sec | 100 reqs | http://127.0.0.1:8000/wait | sync_get_all
196.06 reqs / sec | 100 reqs | http://127.0.0.1:8000/wait | thread_pool
```

We can see there is some overhead on the server-side. Indeed, we should have $100 / 0.05 = 2000$ requests per second if everything worked without any friction.

## `async` routes 

There is another way to declare a route with FastAPI, using the `asyncio` keywords.

```python
# server.py

import asyncio
...

@app.get("/asyncwait")
async def asyncwait():
    duration = 0.05
    await asyncio.sleep(duration)
    return {"duration": duration}
```

Now just add this route to the client:

```python, hl_lines=4
# client.py

if __name__ == "__main__":
    urls = ["http://127.0.0.1:8000/wait", "http://127.0.0.1:8000/asyncwait"]
    run_bench(10, [sync_get_all, thread_pool], urls)
```

And run the benchmark:

```
15.66 reqs / sec | 100 reqs | http://127.0.0.1:8000/wait | sync_get_all
195.41 reqs / sec | 100 reqs | http://127.0.0.1:8000/wait | thread_pool
15.52 reqs / sec | 100 reqs | http://127.0.0.1:8000/asyncwait | sync_get_all
208.06 reqs / sec | 100 reqs | http://127.0.0.1:8000/asyncwait | thread_pool
```

We see a small improvement. But isn't asyncio supposed to be very performant? And [Uvicorn](https://www.uvicorn.org/) is based on [uvloop](https://github.com/MagicStack/uvloop), described as:

> Ultra fast asyncio event loop. 

Maybe the overhead comes from the client? Threadpools maybe?

## Drinking the `asyncio` kool-aid

To check this, we're going to implement a fully-asynchronous client. This is a bit more _involved_. Yes, this means `async`s and `await`s. I know you secretly enjoy these.

Just do `pip install aiohttp`, then:

```python
# client.py

import asyncio
...
import aiohttp

...


async def aget(session, url):
    async with session.get(url) as response:
        assert response.status == 200
        json = await response.json()
        return json


async def gather_limit(n_workers, *tasks):
    semaphore = asyncio.Semaphore(n_workers)

    async def sem_task(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*(sem_task(task) for task in tasks))


async def aget_all(url, n, n_workers=None):
    limit = n_workers or n
    async with aiohttp.ClientSession() as session:
        result = await gather_limit(limit, *[aget(session, url) for _ in range(n)])
        return result


def async_main(url, n):
    return asyncio.run(aget_all(url, n))
```

We also add this function to the benchmark. Let's also add a benchmark with 1000 request, just for async methods.

```python, hl_lines=5 7
# client.py

if __name__ == "__main__":
    urls = ["http://127.0.0.1:8000/wait", "http://127.0.0.1:8000/asyncwait"]
    funcs = [sync_get_all, thread_pool, async_main]
    run_bench(100, funcs, urls)
    run_bench(1000, [thread_pool, async_main], urls)
```

The results can be suprising:

```
15.84 reqs / sec | 100 reqs | http://127.0.0.1:8000/wait | sync_get_all
191.74 reqs / sec | 100 reqs | http://127.0.0.1:8000/wait | thread_pool
187.36 reqs / sec | 100 reqs | http://127.0.0.1:8000/wait | async_main
15.69 reqs / sec | 100 reqs | http://127.0.0.1:8000/asyncwait | sync_get_all
217.35 reqs / sec | 100 reqs | http://127.0.0.1:8000/asyncwait | thread_pool
666.23 reqs / sec | 100 reqs | http://127.0.0.1:8000/asyncwait | async_main
234.24 reqs / sec | 1000 reqs | http://127.0.0.1:8000/wait | thread_pool
222.16 reqs / sec | 1000 reqs | http://127.0.0.1:8000/wait | async_main
316.08 reqs / sec | 1000 reqs | http://127.0.0.1:8000/asyncwait | thread_pool
1031.05 reqs / sec | 1000 reqs | http://127.0.0.1:8000/asyncwait | async_main
```

It appears that the bottleneck was indeed on the client-side! When both sides are asynchronous - and there is a lot of IO - the speed is impressive!

# CPU-bound workloads

This is all great, until some heavy computation is required. We refer to these as _CPU-bound_ workloads, as opposed to _IO-bound_. Inspired by the legendary David Beazley's [live coding](https://youtu.be/MCs5OvhV9S4), we are going to use a naive implementation of the Fibonacci sequence to perform heavy computations.

```python
# server.py

...

def fibo(n):
    if n < 2:
        return 1
    else:
        return fibo(n - 1) + fibo(n - 2)


@app.get("/fib/{n}")
def fib(n: int):
    return {"fib": fibo(n)}
```

Now, when I open two terminals, running `curl -I http://127.0.0.1:8000/fib/42` in one and `python client.py` in the other, we see the following results:

```
8.75 reqs / sec | 100 reqs | http://127.0.0.1:8000/wait | sync_get_all
54.94 reqs / sec | 100 reqs | http://127.0.0.1:8000/wait | thread_pool
60.64 reqs / sec | 100 reqs | http://127.0.0.1:8000/wait | async_main
9.52 reqs / sec | 100 reqs | http://127.0.0.1:8000/asyncwait | sync_get_all
53.02 reqs / sec | 100 reqs | http://127.0.0.1:8000/asyncwait | thread_pool
46.81 reqs / sec | 100 reqs | http://127.0.0.1:8000/asyncwait | async_main
72.87 reqs / sec | 1000 reqs | http://127.0.0.1:8000/wait | thread_pool
122.97 reqs / sec | 1000 reqs | http://127.0.0.1:8000/wait | async_main
72.36 reqs / sec | 1000 reqs | http://127.0.0.1:8000/asyncwait | thread_pool
51.73 reqs / sec | 1000 reqs | http://127.0.0.1:8000/asyncwait | async_main
```

It's not that bad, but a bit disappointing. Indeed, we have 20x less throughput for the originally most performant one (`asyncwait` route x `async_main` client). 

What's happening here ? In python, there is a [Global Interpreter Lock](https://wiki.python.org/moin/GlobalInterpreterLock) (GIL). If one request takes a very long time to be processed with high-CPU activity, in the meantime other requests cannot be processed as quickly: priority is given to the computations. We will see [later](#gunicorn-and-multiprocessing) how to take care of this.

For now, we try nested recursive concurrency. Let's add:

```python
# server.py

...


async def afibo(n):
    if n < 2:
        return 1
    else:
        fib1 = await afibo(n - 1)
        fib2 = await afibo(n - 2)
        return fib1 + fib2


@app.get("/asyncfib/{n}")
async def asyncfib(n: int):
    res = await afibo(n)
    return {"fib": res}
```

Let's also add a [timing middleware](https://fastapi.tiangolo.com/tutorial/middleware/) to our FastAPI app:

```python
# server.py
...
from fastapi import FastAPI, Request

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response
```

Now let's test the speed:

```bash
curl -D - http://127.0.0.1:8000/fib/30
```
``` hl_lines=5
HTTP/1.1 200 OK
server: uvicorn
content-length: 15
content-type: application/json
x-process-time: 0.17467308044433594

{"fib":1346269}⏎                                                                                     
```

And with async:

```bash
curl -D - http://127.0.0.1:8000/asyncfib/30
```
``` hl_lines=5
HTTP/1.1 200 OK
server: uvicorn
content-length: 15
content-type: application/json
x-process-time: 0.46001315116882324

{"fib":1346269}⏎                                                                                     
```

It's not that bad for $2^{30}$ overhead. But we see here a limitation of threads in Python: the [same code in Julia](https://julialang.org/blog/2019/07/multithreading/) would lead to a speed-up (using parallelism)!


# Gunicorn and multiprocessing

So far we've used FastAPI with Uvicorn. The latter can be [run with Gunicorn](https://www.uvicorn.org/#running-with-gunicorn). Gunicorn forks a base process into `n` worker processes, and each worker is managed by Uvicorn (with the asynchronous uvloop). Which means:
- Each worker is concurrent
- The worker pool implements parallelism

This way, we can have the best of both worlds: concurrency (multithreading) and parallelism (multiprocessing).

Let's try this with the last setup, when we ran the benchmark while asking for the 42th Fibonacci number:
```bash
pip install gunicorn
```

```bash
gunicorn server:app -w 2 -k uvicorn.workers.UvicornWorker --reload
```

we get the following results:

```
19.02 reqs / sec | 100 reqs | http://127.0.0.1:8000/wait | sync_get_all
216.84 reqs / sec | 100 reqs | http://127.0.0.1:8000/wait | thread_pool
223.52 reqs / sec | 100 reqs | http://127.0.0.1:8000/wait | async_main
18.80 reqs / sec | 100 reqs | http://127.0.0.1:8000/asyncwait | sync_get_all
400.12 reqs / sec | 100 reqs | http://127.0.0.1:8000/asyncwait | thread_pool
208.68 reqs / sec | 100 reqs | http://127.0.0.1:8000/asyncwait | async_main
241.06 reqs / sec | 1000 reqs | http://127.0.0.1:8000/wait | thread_pool
311.40 reqs / sec | 1000 reqs | http://127.0.0.1:8000/wait | async_main
433.80 reqs / sec | 1000 reqs | http://127.0.0.1:8000/asyncwait | thread_pool
1275.48 reqs / sec | 1000 reqs | http://127.0.0.1:8000/asyncwait | async_main   
```

Which is on par (if not a bit better!) than with a single Uvicorn process

The final files (client and server) are available as a [github gist](https://gist.github.com/horaceg/2afe33aff9ba28c36757f52f13edd298)


# Further resources

I wholeheartedly recomment this amazing [live-coding session](https://youtu.be/MCs5OvhV9S4) by David Beazley. Maybe you can google [websockets](https://en.wikipedia.org/wiki/WebSocket) first, just to get that they open a bi-directional channel between client and server.

You can also read [this detailed answer](https://stackoverflow.com/a/60644649/9162021) from stackoverflow to grasp differences between concurrency and parallelism in python.
