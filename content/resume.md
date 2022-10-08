+++
title = "The responsive markdown resume"
date = 2022-02-24

[taxonomies]
categories = ["web"]
tags = ["HTML", "programming", "CSS", "javascript", "resume", "CV"]
+++

I decided to do my resume since I haven't done it in a while. My old one was based on the latex template _moderncv_. But this time, I wanted to have one _for the web_ that looks good on desktop and mobile, as well as paper (pdf).

TL;DR: [horaceg.xyz](https://horaceg.xyz)

## Inspiration

I saw a [blog post](https://blog.chmd.fr/editing-a-cv-in-markdown-with-pandoc.html) by Christophe-Marie Duquesne that seemed to exactly fit the bill. It is even inspired by _moderncv_ !

So I ~~copied~~ took insipiration from it and wrote my own experience. However, it didn't display well on mobile since the left section overflows to the center one.

Since we are in 2022, mobile first and all, I modified it to look good on mobile as well as _desktop_ and _pdf_.

## Width and overflow

The first thing I had to modify was chaning the body `width` property to `max-width`:

```css
body {
    ...
    max-width: 900px
}
```

Then, I had to manage the overflow of the left section on mobile:

```css
dt {
    ...
    overflow-wrap: break-word;
}
```

This fixed the first issue: now one could somewhat read it on mobile !

## Collapsible items

Still, it was hardly readable since there's a lot of text that spans many paragraphs for each item. So here I had the idea that I need to make the individual items collapsible.

Fortunately, there is exactly an HTML element to do this: `<details>` & `<summary>` ! There is no specific markdown syntax to deal with these, so I had to put it as-is in the `.md` file (having tried and failed to write a custom pandoc filter).

```markdown
2019 - now
:   <details open><summary>*Data scientist* in R&D at **Deepki** (Paris, France)</summary>
    
    Data science improving energy efficiency of buildings.
    ...
    </details>
```

## Collapsed on mobile by default

This looks good, but there's still an improvement possible. I want to show only the summary on mobile, and show the full, extended version on desktop.

This is where this bit of `javascript` helped: 

```js
window.addEventListener("load", function () {
  var elements = document.getElementsByTagName("details");
  for (let e of elements) {
    if (window.innerWidth < 500) {
      e.open = false;
    } else {
      e.open = true;
    }
  }
})
```

On the `load` event, this script toggles the `open` attribute of all the `details` elements based on the screen width.

## Showing a preview on collapsed

There's one thing left to be desired: since it does not look like a button, users won't necessrily understand that collapsed sections are clickable.
One could make it more obvious by styling the summary like a button, but I liked the clean simple design.

So I added this snippet of css:

```css
summary::after {
  content: "\a" attr(preview);
  white-space: pre;
  opacity: 0.5;
}

details[open] > summary::after {
  content: none;
}
```

On collapsed, it shows the `preview` attribute of the `summary` element with half opacity, preceded by a newline `"\a"`.

Now it works but I still have to manually fill the preview attributes, which is boring and error-prone in case I change items.

## Setting the element automatically

```js, hl_lines=5-13
window.addEventListener("load", function () {
  var elements = document.getElementsByTagName("details");
  for (let e of elements) {
      ...
    e.children[0].setAttribute(
      "preview",
      e.outerHTML
        .split("<p>")[1]
        .split(" ")
        .slice(0, 4)
        .join(" ")
        .replace("amp;", "") + "..."
    );
  }
});
```

Now, the preview attribute of all details elements contains the first 4 words of the following section, plus "..." appended at the end.


## Building HTML and pdf documents

To build in html, I need to include the javascript in the header of the html file. Since pandoc uses `wkhtmltopdf`, and the latter puts large margins everywhere by default, I need to specify the margins at zero for them to be handled by the css.

In `build.bash`: 

```bash
#! /bin/bash

set -e

date

pandoc -s --from markdown --to html \
    -H <(echo "<script>" $(cat script.js) "</script>") \
    -c style.css -o resume.html resume.md

pandoc -s --from markdown --to html \
    -V margin-top=0 -V margin-left=0 -V margin-right=0 -V margin-bottom=0 \
    -V papersize=letter \
    -c style.css -o horace_guy.pdf resume.md

```

When I edit the document, I use `entr` to watch for files in the folder and build on change. In `watch.bash`:
```bash
#! /bin/bash

set -e

ls *.{md,css,js} | entr ./build.bash
```

The dev workflow is now: `./watch.bash`.

## Putting it online

Well, we need to put online an html file, a css file and a pdf file. I use [sitejs](https://sitejs.org/) with `site push` to put it online on a $5/month Linode box.

## Wrap-up

All in all, I am happy with this new resume. I need to focus more on the content now that I have the template.
All input files are in the [github repo](https://github.com/horaceg/resume)
