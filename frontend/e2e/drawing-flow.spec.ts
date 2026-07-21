import { expect, test } from '@playwright/test';

const png = Buffer.from(
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
  'base64',
);

const corsHeaders = {
  'access-control-allow-origin': '*',
  'access-control-allow-headers': 'content-type',
  'access-control-expose-headers': 'x-execution-token',
};

test('renders, locks controls, saves the rendered code, and invalidates edited code', async ({ page }) => {
  let releaseExecution: () => void = () => undefined;
  const executionGate = new Promise<void>((resolve) => {
    releaseExecution = resolve;
  });
  let storedRequest: Record<string, unknown> | undefined;

  await page.route('https://analytics.rashik.sh/**', (route) => route.fulfill({ status: 204 }));
  await page.route('http://localhost:8000/**', async (route) => {
    const request = route.request();
    if (request.method() === 'OPTIONS') {
      await route.fulfill({ status: 204, headers: corsHeaders });
      return;
    }

    if (request.url().endsWith('/generate_code')) {
      await route.fulfill({
        status: 200,
        headers: { ...corsHeaders, 'content-type': 'application/json' },
        body: JSON.stringify({ code: 'from artcanvas import ArtCanvas\ncanvas = ArtCanvas()\ncanvas.save()' }),
      });
      return;
    }

    if (request.url().endsWith('/run_code')) {
      await executionGate;
      await route.fulfill({
        status: 200,
        headers: {
          ...corsHeaders,
          'content-type': 'image/png',
          'x-execution-token': 'signed-execution-token',
        },
        body: png,
      });
      return;
    }

    if (request.url().endsWith('/store_code')) {
      storedRequest = request.postDataJSON() as Record<string, unknown>;
      await route.fulfill({
        status: 200,
        headers: { ...corsHeaders, 'content-type': 'application/json' },
        body: JSON.stringify({ success: true }),
      });
      return;
    }

    await route.abort();
  });

  await page.goto('/');
  await page.getByRole('button', { name: 'Show Docs' }).click();
  await expect(page.getByRole('dialog')).toContainText('canvas.move_brush_to(x, y)');
  await page.getByRole('button', { name: 'Close documentation' }).click();

  const prompt = page.locator('input[type="text"]');
  await prompt.fill('A precise red line');
  await page.getByRole('button', { name: 'Generate Code' }).click();

  await expect(prompt).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Reset' })).toBeDisabled();
  await expect(page.getByRole('button', { name: 'Save to public gallery' })).toBeDisabled();

  releaseExecution();
  await expect(page.getByAltText('Rendered drawing')).toBeVisible();
  const saveButton = page.getByRole('button', { name: 'Save to public gallery' });
  await expect(saveButton).toBeEnabled();
  await saveButton.click();
  await expect(page.getByText('Drawing saved successfully')).toBeVisible();
  expect(storedRequest).toMatchObject({
    prompt: 'A precise red line',
    artType: 'drawing',
    executionToken: 'signed-execution-token',
  });

  await page.locator('textarea').fill('from artcanvas import ArtCanvas\n# changed after rendering');
  await expect(page.getByAltText('Rendered drawing')).toHaveCount(0);
  await expect(saveButton).toBeDisabled();
  await page.screenshot({ path: 'test-results/drawing-flow.png', fullPage: true });
});
