"""Optional UI check: python tests/browser_smoke.py output/viewer.html.

Requires playwright plus Chromium. Set CHROMIUM_EXECUTABLE to use a system
browser; otherwise install the Playwright browser with playwright install chromium.
"""

import argparse
import os
from pathlib import Path


def main():
    from playwright.sync_api import sync_playwright
    parser=argparse.ArgumentParser()
    parser.add_argument('html',type=Path)
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    with sync_playwright() as p:
        options={"headless":True,"args":["--no-sandbox"]}
        if os.environ.get('CHROMIUM_EXECUTABLE'):
            options['executable_path']=os.environ['CHROMIUM_EXECUTABLE']
        browser=p.chromium.launch(**options)
        page=browser.new_page(viewport={"width":1280,"height":960},reduced_motion='reduce')
        errors=[]; network=[]
        page.on('pageerror',lambda e:errors.append(str(e)))
        page.on('request',lambda r:network.append(r.url))
        # set_content also works in enterprise browsers that forbid file: URLs.
        # This injects the exact exported document, without a server or network.
        page.set_content(args.html.read_text(encoding='utf-8'))
        assert page.locator('#degree').inner_text()=='0.0°'
        page.locator('#angle').fill('0'); page.locator('#angle').dispatch_event('input')
        assert page.locator('#degree').inner_text()=='-15.0°'
        page.locator('#edges').check(); page.locator('#envelopes').check()
        before=page.locator('#degree').inner_text()
        page.locator('#play').click(); page.wait_for_timeout(250); page.locator('#play').click()
        assert before!=page.locator('#degree').inner_text()
        page.locator('#scan').select_option('target_candidate')
        assert '干渉' in page.locator('#status').inner_text()
        page.locator('#scan').select_option('prototype_candidate')
        page.locator('#envelopes').uncheck(); page.locator('#edges').uncheck()
        page.mouse.move(500,300); page.mouse.down(); page.mouse.move(560,340); page.mouse.up()
        page.locator('#reset').click()
        if args.screenshot:
            page.screenshot(path=str(args.screenshot),full_page=True)
        page.set_viewport_size({"width":390,"height":844})
        assert page.evaluate('document.documentElement.scrollWidth <= window.innerWidth')
        page.locator('#scan').select_option('prototype_full')
        assert page.locator('#angle').get_attribute('max')=='360'
        assert not errors, errors
        assert not network, network
        browser.close()
    print('Browser smoke passed: controls, playback, camera, mobile layout, no network requests or JS errors.')


if __name__=='__main__':
    main()
