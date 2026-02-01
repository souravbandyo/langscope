import { test, expect } from '@playwright/test'

test.describe('Rankings Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/rankings')
  })

  test('displays the rankings title', async ({ page }) => {
    await expect(page.getByRole('heading', { name: 'Rankings' })).toBeVisible()
  })

  test('has domain filter dropdown', async ({ page }) => {
    const domainSelect = page.locator('select').first()
    await expect(domainSelect).toBeVisible()
    
    // Check some options exist
    await expect(domainSelect.locator('option')).toHaveCount(6)
  })

  test('has dimension filter dropdown', async ({ page }) => {
    const selects = page.locator('select')
    await expect(selects).toHaveCount(2)
  })

  test('displays leaderboard table', async ({ page }) => {
    // Check table headers
    await expect(page.getByText('Rank')).toBeVisible()
    await expect(page.getByText('Model')).toBeVisible()
    await expect(page.getByText('Provider')).toBeVisible()
    await expect(page.getByText('Rating')).toBeVisible()
  })

  test('displays medal icons for top 3', async ({ page }) => {
    await expect(page.getByText('🥇')).toBeVisible()
    await expect(page.getByText('🥈')).toBeVisible()
    await expect(page.getByText('🥉')).toBeVisible()
  })

  test('can change domain filter', async ({ page }) => {
    const domainSelect = page.locator('select').first()
    await domainSelect.selectOption('code_generation')
    
    // Page should still be functional
    await expect(page.getByRole('heading', { name: 'Rankings' })).toBeVisible()
  })
})
