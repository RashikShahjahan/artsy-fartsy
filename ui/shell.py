"""Shared navigation and document shell for Artsy workspaces."""
import html


def navigation(current):
    links = [('studio', '/studio', 'Studio'), ('review', '/review', 'Prompt review'),
             ('candidates', '/candidates', 'SFT candidates'),
             ('runs', '/runs', 'Runs'),
             ('comparisons', '/comparisons', 'Eval comparison'),
             ('preferences', '/preferences', 'Blind ranking'),
             ('rl-prompts', '/rl/prompts', 'RL prompts'),
             ('rl-rollouts', '/rl/rollouts', 'RL rollouts')]
    return '<header class="app-header"><a class="brand" href="/">Artsy<span class="brand-dot">.</span></a><nav aria-label="Main navigation">' + ''.join(
        f'<a href="{url}"' + (' aria-current="page"' if key == current else '') + f'>{label}</a>'
        for key, url, label in links) + '</nav><span class="prototype">Drawing workspace</span></header>'


def review_page(content, current):
    """Wrap explicit review markup; do not parse or rewrite its HTML."""
    title = {'review': 'Prompt review', 'candidates': 'SFT candidates'}[current]
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>Artsy · {title}</title><link rel="stylesheet" href="/style.css"></head>'
            f'<body class="{current}-page">' + navigation(current) + content + '</body></html>')


def empty_page(current, title, message):
    return review_page(f'<header class="workspace-header"><h1>{html.escape(title)}</h1></header>'
                       f'<main class="review-workspace"><div class="panel ui-panel empty-state"><p>{html.escape(message)}</p>'
                       '<a href="/studio">Return to Studio</a></div></main>', current)
