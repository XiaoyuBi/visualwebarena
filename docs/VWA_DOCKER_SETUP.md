# VWA Docker Setup Guide

Follow the steps below to set up your instance and connection.

---

## 1. Update `docker-compose.yml`

After connecting to your instance, open the file `classifieds_docker_compose/docker-compose.yml` and set the `CLASSIFIEDS` environment variable to your site URL:

```yaml
CLASSIFIEDS=http://<your-server-hostname>:9980/
```

---

## 2. Define Your Hostname

**Edit this line with your server's hostname before proceeding:**

```bash
HOSTNAME="<your-server-hostname>"
```

---

## 3. Start Required Containers

```bash
docker start shopping
docker start forum
docker start kiwix33
cd ~/classifieds_docker_compose
```

---

## 4. Build and Start Docker Compose

```bash
docker compose up --build -d
```

---

## 5. VWA Setup

Initialize the `osclass` database with:

```bash
docker exec classifieds_db mysql -u root -ppassword osclass -e 'source docker-entrypoint-initdb.d/osclass_craigslist.sql'
```

Update Magento base URL and flush cache:

```bash
docker exec shopping /var/www/magento2/bin/magento setup:store-config:set --base-url="http://${HOSTNAME}:7770"
docker exec shopping mysql -u magentouser -pMyPassword magentodb -e "UPDATE core_config_data SET value=\"http://${HOSTNAME}:7770/\" WHERE path = \"web/secure/base_url\";"
docker exec shopping /var/www/magento2/bin/magento cache:flush
```

---

## 6. Disable Magento Re-Indexing of Products

Run the following commands to set the Magento indexers to schedule mode:

```bash
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalogrule_product
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalogrule_rule
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalogsearch_fulltext
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_category_product
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule customer_grid
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule design_config_grid
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule inventory
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_category
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_attribute
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule catalog_product_price
docker exec shopping /var/www/magento2/bin/magento indexer:set-mode schedule cataloginventory_stock
```

---
 
**Your VWA Docker environment should now be set up!**