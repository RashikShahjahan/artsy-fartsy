"""Shared navigation and document shell for all Artsy workspaces."""
import html
import re


def navigation(current):
    links = [('studio', '/', 'Studio'), ('review', '/review', 'Prompt review'),
             ('candidates', '/candidates', 'SFT candidates'),
             ('rl', '/rl', 'RL monitor')]
    return '<header class="app-header"><a class="brand" href="/">Artsy<span class="brand-dot">.</span></a><nav aria-label="Main navigation">' + ''.join(
        f'<a href="{url}"' + (' aria-current="page"' if key == current else '') + f'>{label}</a>'
        for key, url, label in links) + '</nav><span class="prototype">Drawing workspace</span></header>'


def review_page(page, current):
    """Keep the existing review controls and behavior in the common app shell."""
    page = re.sub(r'<style>.*?</style>', '', page, flags=re.S)
    page = re.sub(r'^.*?(?=<header>)', '', page, count=1, flags=re.S)
    page = page.replace('<header>', '<header class="workspace-header">', 1)
    page = page.replace('Artsy benchmark · Review before evaluation', 'Prompt review')
    page = page.replace('Artsy · Prompt review', 'Prompt review')
    page = page.replace('Artsy · Manual safety inspector', 'Safety prompt review')
    page = page.replace('Artsy · Full SFT data inspector', 'SFT candidates')
    page = page.replace('<main>', '<main class="review-workspace">', 1)
    page = page.replace('</html>', '')
    for ident, label in [('search', 'Search prompts or candidates'), ('reviewer', 'Reviewer name'),
                         ('slice', 'Prompt category'), ('filter', 'Review decision'),
                         ('automatic', 'Generation outcome'), ('decision', 'Manual decision')]:
        page = page.replace(f'id="{ident}"', f'id="{ident}" aria-label="{label}"')
    title = {'review': 'Prompt review', 'candidates': 'SFT candidates'}[current]
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Artsy · {title}</title><link rel="stylesheet" href="/style.css"></head>'
            f'<body class="{current}-page">' + navigation(current) + page + '</body></html>')


def empty_page(current, title, message):
    return review_page(f'<header><h1>{html.escape(title)}</h1></header>'
                       f'<main><div class="panel empty-state"><p>{html.escape(message)}</p>'
                       '<a href="/">Return to Studio</a></div></main>', current)
