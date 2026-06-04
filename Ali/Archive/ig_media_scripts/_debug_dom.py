"""Debug script — opens one post and prints what img/video elements exist in the DOM."""
from pathlib import Path
from playwright.sync_api import sync_playwright

USER_DATA_DIR = str((Path(__file__).parent / "_debug_profile").resolve())
SHORTCODE = "CSjPl6KLiZg"  # a known IMAGE post

with sync_playwright() as p:
    ctx = p.chromium.launch_persistent_context(
        USER_DATA_DIR,
        headless=False,
        viewport={"width": 1280, "height": 900},
        args=["--disable-blink-features=AutomationControlled"],
    )
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    page.goto(f"https://www.instagram.com/p/{SHORTCODE}/", wait_until="networkidle", timeout=30000)
    page.wait_for_timeout(3000)

    result = page.evaluate("""
    () => {
        const out = {};

        // 1. What top-level tags exist near post content?
        out.has_article = !!document.querySelector('article');
        out.has_main    = !!document.querySelector('main');

        // 2. All img elements — src snippet, size, alt snippet
        out.all_imgs = [...document.querySelectorAll('img')].map(img => ({
            src_head : (img.src||'').slice(0,80),
            srcset_has: !!img.srcset,
            w: img.naturalWidth,
            h: img.naturalHeight,
            alt: (img.alt||'').slice(0,40),
        }));

        // 3. Specifically images inside article (if any)
        out.article_imgs = [...document.querySelectorAll('article img')].map(img => ({
            src_head: (img.src||'').slice(0,80),
            w: img.naturalWidth, h: img.naturalHeight,
        }));

        // 4. Specifically images inside main
        out.main_imgs = [...document.querySelectorAll('main img')].map(img => ({
            src_head: (img.src||'').slice(0,80),
            w: img.naturalWidth, h: img.naturalHeight,
        }));

        // 5. Any div with role=dialog (post modal)
        out.dialog_imgs = [...document.querySelectorAll('[role="dialog"] img')].map(img => ({
            src_head: (img.src||'').slice(0,80),
            w: img.naturalWidth, h: img.naturalHeight,
        }));

        // 6. Look for large images anywhere (likely the post image)
        out.large_imgs = [...document.querySelectorAll('img')].filter(
            img => img.naturalWidth > 300 && img.naturalHeight > 300
        ).map(img => ({
            src_head: (img.src||'').slice(0,100),
            srcset_head: (img.srcset||'').slice(0,100),
            w: img.naturalWidth, h: img.naturalHeight,
            alt: (img.alt||'').slice(0,60),
            parent_tag: img.parentElement ? img.parentElement.tagName : '',
            grandparent_tag: img.parentElement && img.parentElement.parentElement
                             ? img.parentElement.parentElement.tagName : '',
        }));

        return out;
    }
    """)

    import json
    print(f"has_article: {result['has_article']}")
    print(f"has_main:    {result['has_main']}")
    print(f"total imgs on page: {len(result['all_imgs'])}")
    print(f"article imgs: {len(result['article_imgs'])}")
    print(f"main imgs:    {len(result['main_imgs'])}")
    print(f"dialog imgs:  {len(result['dialog_imgs'])}")
    print(f"\nLARGE images (>300x300): {len(result['large_imgs'])}")
    for img in result['large_imgs']:
        print(f"  {img['w']}x{img['h']}  parent={img['parent_tag']}/{img['grandparent_tag']}  alt={img['alt']!r}")
        print(f"    src:    {img['src_head']}")
        print(f"    srcset: {img['srcset_head']}")

    ctx.close()