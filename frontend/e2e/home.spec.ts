import { test, expect } from '@playwright/test'

test.describe('Home Page', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto('/')
  })

  test('displays the LangScope logo', async ({ page }) => {
    await expect(page.getByText('LANGSCOPE')).toBeVisible()
  })

  test('displays the tagline', async ({ page }) => {
    await expect(
      page.getByText('Find the best model for your needs')
    ).toBeVisible()
  })

  test('has a search bar', async ({ page }) => {
    const searchInput = page.getByPlaceholder('Describe your usecase...')
    await expect(searchInput).toBeVisible()
  })

  test('displays trending domains section', async ({ page }) => {
    await expect(page.getByText('Trending Domains')).toBeVisible()
  })

  test('displays domain leaderboard stickies', async ({ page }) => {
    await expect(page.getByText('Domain Leaderboards')).toBeVisible()
  })

  test('can navigate to rankings from trending domain', async ({ page }) => {
    await page.getByText('#Code Generation').click()
    await expect(page).toHaveURL(/\/rankings/)
  })
})

test.describe('Navigation', () => {
  test('sidebar navigation works', async ({ page }) => {
    await page.goto('/')

    // Navigate to Rankings
    await page.getByRole('link', { name: 'Rankings' }).click()
    await expect(page).toHaveURL('/rankings')
    await expect(page.getByText('Rankings')).toBeVisible()

    // Navigate to Arena
    await page.getByRole('link', { name: 'Arena' }).click()
    await expect(page).toHaveURL('/arena')
    await expect(page.getByText('Arena Mode')).toBeVisible()

    // Navigate to About
    await page.getByRole('link', { name: 'About' }).click()
    await expect(page).toHaveURL('/about')
    await expect(page.getByText('About LangScope')).toBeVisible()

    // Navigate back Home
    await page.getByRole('link', { name: 'Home' }).click()
    await expect(page).toHaveURL('/')
  })
})
