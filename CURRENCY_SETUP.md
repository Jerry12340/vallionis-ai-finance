# Multi-Currency Setup Guide

## Overview
The website now supports multiple currencies: USD, EUR, and GBP. Users can select their preferred currency and prices will be displayed accordingly.

## Environment Variables Required

Add these environment variables to your `.env` file:

```bash
# USD Prices (Stripe Price IDs)
STRIPE_USD_MONTHLY_PRICE_ID=price_xxxxxxxxxxxxx
STRIPE_USD_LIFETIME_PRICE_ID=price_xxxxxxxxxxxxx

# EUR Prices (Stripe Price IDs)  
STRIPE_EUR_MONTHLY_PRICE_ID=price_xxxxxxxxxxxxx
STRIPE_EUR_LIFETIME_PRICE_ID=price_xxxxxxxxxxxxx

# GBP Prices (Stripe Price IDs) - existing
STRIPE_GBP_MONTHLY_PRICE_ID=price_xxxxxxxxxxxxx
STRIPE_GBP_LIFETIME_PRICE_ID=price_xxxxxxxxxxxxx
```

## Stripe Setup

### 1. Create Price Products in Stripe Dashboard

For each currency, create the following products:

**USD Products:**
- Monthly subscription: $6.99/month
- Lifetime access: $129.99 (one-time)

**EUR Products:**
- Monthly subscription: €5.99/month  
- Lifetime access: €109.99 (one-time)

**GBP Products:**
- Monthly subscription: £4.99/month
- Lifetime access: £99.99 (one-time)

### 2. Get Price IDs

After creating each product in Stripe, copy the Price IDs and add them to your environment variables.

## Features

### Currency Detection
- Currently defaults to GBP
- Can be enhanced with IP geolocation services
- Users can manually select their preferred currency

### Dynamic Pricing
- Prices update automatically when currency is changed
- All forms include the selected currency
- Schema markup updates for SEO

### Supported Currencies
- **USD**: $6.99/month, $129.99 lifetime
- **EUR**: €5.99/month, €109.99 lifetime  
- **GBP**: £4.99/month, £99.99 lifetime

## Future Enhancements

### IP-Based Detection
To automatically detect user's currency based on location:

```python
# Install: pip install requests
import requests

def detect_user_currency():
    try:
        # Get user's IP and country
        response = requests.get('https://ipapi.co/json/')
        data = response.json()
        
        # Map countries to currencies
        currency_map = {
            'US': 'USD',
            'CA': 'USD',  # or CAD if you want to support it
            'GB': 'GBP',
            'DE': 'EUR',
            'FR': 'EUR',
            # Add more countries as needed
        }
        
        return currency_map.get(data.get('country_code'), 'GBP')
    except:
        return 'GBP'  # Default fallback
```

### Additional Currencies
To add more currencies (e.g., CAD, AUD, JPY):

1. Add currency configuration to `CURRENCY_CONFIG`
2. Create corresponding Stripe price products
3. Add environment variables
4. Update the currency selector in templates

## Testing

1. Set up all environment variables
2. Create test products in Stripe test mode
3. Test currency switching on the subscription page
4. Verify payments work in each currency
5. Check that webhook handling works correctly

## Notes

- All prices are configured in the `CURRENCY_CONFIG` dictionary in `app.py`
- The default currency is GBP if detection fails
- Currency selection is stored in form submissions
- Stripe handles currency conversion automatically 