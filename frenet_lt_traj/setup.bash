# curl -o setup.bash http://mlp-api-gateway.srv.deeproute.cn/anyviz/static/install/install.sh && anyviz_host=http://mlp-api-gateway.srv.deeproute.cn/anyviz  sudo -E bash setup.bash && rm setup.bash

# 需要强制关闭上一次的程序
sudo netstat -tunlp | grep 8082 | awk '{print $7}' | awk -F'/' '{print $1}' | xargs sudo kill -9

echo api-anyviz ${anyviz_host}/static/api-anyviz

curl -o api-anyviz ${anyviz_host}/static/api-anyviz

mv ./api-anyviz /bin/anyviz
chmod +x /bin/anyviz
anyviz -l &