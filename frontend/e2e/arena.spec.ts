import { test, expect } from '@playwright/test'

test.describe('Arena Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/arena')
  })

  test('displays arena title and subtitle', async ({ page }) => {
    await expect(page.getByText('Arena Mode')).toBeVisible()
    await expect(
      page.getByText('Compare models head-to-head')
    ).toBeVisible()
  })

  test('shows setup form initially', async ({ page }) => {
    await expect(page.getByText('Start a New Session')).toBeVisible()
    await expect(page.getByText('Select Domain:')).toBeVisible()
    await expect(page.getByText('Number of Battles:')).toBeVisible()
  })

  test('has start session button', async ({ page }) => {
    const startButton = page.getByRole('button', { name: 'Start Arena Session' })
    await expect(startButton).toBeVisible()
  })

  test('can select domain', async ({ page }) => {
    const domainSelect = page.locator('select').first()
    await domainSelect.selectOption('mathematical_reasoning')
    
    await expect(domainSelect).toHaveValue('mathematical_reasoning')
  })

  test('starts battle when clicking start button', async ({ page }) => {
    await page.getByRole('button', { name: 'Start Arena Session' }).click()
    
    // Should transition to battle view
    await expect(page.getByText('Battle 1 of 5')).toBeVisible()
  })

  test('battle view shows prompt and responses', async ({ page }) => {
    await page.getByRole('button', { name: 'Start Arena Session' }).click()
    
    await expect(page.getByText('Prompt')).toBeVisible()
    await expect(page.getByText('Response A')).toBeVisible()
    await expect(page.getByText('Response B')).toBeVisible()
  })

  test('can navigate through battles', async ({ page }) => {
    await page.getByRole('button', { name: 'Start Arena Session' }).click()
    
    // Click next battle
    await page.getByRole('button', { name: 'Next Battle →' }).click()
    await expect(page.getByText('Battle 2 of 5')).toBeVisible()
  })

  test('completes session after all battles', async ({ page }) => {
    await page.getByRole('button', { name: 'Start Arena Session' }).click()
    
    // Go through all 5 battles
    for (let i = 0; i < 4; i++) {
      await page.getByRole('button', { name: 'Next Battle →' }).click()
    }
    
    // Final battle should show "Finish Session"
    await page.getByRole('button', { name: 'Finish Session' }).click()
    
    // Should show results
    await expect(page.getByText('Session Complete!')).toBeVisible()
  })
})
