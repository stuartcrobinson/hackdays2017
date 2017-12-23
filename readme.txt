presentation:

http://slides.com/stuartcrobinson/deck#/

web app in hackdays2017/web/attempt2/vuex-test-01

run on ec2 large



commands to restart solr when it inevitably goes down:

./bin/solr start -c -p 8983 -s example/cloud/node1/solr

./bin/solr start -c -p 7574 -s example/cloud/node2/solr -z localhost:9983


command to restart vue:

in hackdays2017/web/attempt2/vuex-test-01

nohup npm run dev >/dev/null 2>&1 &


to start elasticsearch (used for typeahead)

sudo -u bronto sh /usr/local/bronto/commerce-es/bin/elasticsearch -d

i think python3 app.py to run python backend