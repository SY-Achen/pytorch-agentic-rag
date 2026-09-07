import json, time
import paramiko
from pathlib import Path

HOST = '47.114.33.146'
USER = 'root'
PASSWORD = 'Ll2112724465'
LOCAL_DATASET = Path(r'C:\Users\Administrator\rag_agent\50_eval_dataset.json')
REMOTE_DIR = '/opt/rag_agent'
REMOTE_DATASET = REMOTE_DIR + '/50_eval_dataset.json'
REMOTE_RUNNER = REMOTE_DIR + '/run_eval_strict.py'

runner = r'''import json, time
import requests

BASE = "http://127.0.0.1:8000"
with open("/opt/rag_agent/50_eval_dataset.json", encoding="utf-8") as f:
    cases = json.load(f)
login = requests.post(BASE + "/api/login", json={"username":"zhangsan","password":"123"}, timeout=20)
login.raise_for_status()
token = login.json()["token"]
headers = {"Authorization": "Bearer " + token}

def call(msg):
    start = time.perf_counter()
    r = requests.post(BASE + "/api/chat", json={"message":msg, "stream":True}, headers=headers, stream=True, timeout=90)
    route = None; sources = 0; reply = ""; cache = False; events=[]
    for raw in r.iter_lines():
        if not raw: continue
        line = raw.decode("utf-8", "replace")
        if not line.startswith("data: "): continue
        try: e = json.loads(line[6:])
        except Exception: continue
        typ = e.get("type"); events.append(typ)
        if typ == "chunk": reply += e.get("text", "")
        elif typ == "sources": sources = len(e.get("sources", []))
        elif typ in ("done", "complete"):
            s = e.get("summary") or e.get("payload") or {}
            route = s.get("tool") or route
            sources = s.get("sources_count", sources)
            cache = cache or bool(s.get("cache_hit"))
    return {"route":route, "sources":int(sources or 0), "reply":reply, "cache":cache, "elapsed_ms":round((time.perf_counter()-start)*1000,1), "events":events}

results=[]
def one(cid, cat, expected, msg, keywords=None):
    x=call(msg); keywords=keywords or []
    result={"id":cid,"category":cat,"expected":expected,"actual":x["route"],"route_ok":x["route"]==expected,"sources":x["sources"],"keywords":keywords,"keywords_all":all(k.lower() in x["reply"].lower() for k in keywords),"cache":x["cache"],"reply":x["reply"],"elapsed_ms":x["elapsed_ms"],"events":x["events"]}
    results.append(result)
    print("[%s] %s expected=%s actual=%s sources=%s kw=%s cache=%s ms=%s" % (cid,cat,expected,x["route"],x["sources"],result["keywords_all"],x["cache"],x["elapsed_ms"]))

for c in [x for x in cases if 1 <= x["id"] <= 25]: one(c["id"],"financial", "KNOWLEDGE_SEARCH", c["q"]+" [strict-fin-20260907]", c.get("must_contain",[]))
for c in [x for x in cases if 26 <= x["id"] <= 35]: one(c["id"],"calculator", "CALCULATE", c["q"]+" [strict-calc-20260907]", c.get("must_contain",[]))
for c in [x for x in cases if 36 <= x["id"] <= 45]: one(c["id"],"general", c["expected_tool"], c["q"]+" [strict-general-20260907]", c.get("must_contain",[]))
for c in [x for x in cases if 46 <= x["id"] <= 50]: one(c["id"],"injection", "INJECT_GUARD", c["q"]+" [strict-injection-20260907]", c.get("must_contain",[]))

memory_cases=[
("m1","MEM_ALPHA_731"),("m2","MEM_BETA_842"),("m3","MEM_GAMMA_953"),("m4","MEM_DELTA_164"),("m5","MEM_EPSILON_275")]
for cid,value in memory_cases:
    call("请记住本轮测试编号是 %s，只记录这个值，不需要检索金融知识。 [%s]"%(value,cid))
    x=call("请问本轮测试编号是什么？ [%s]"%cid)
    ok=value.lower() in x["reply"].lower()
    results.append({"id":cid,"category":"memory","expected":"memory","actual":x["route"],"memory_ok":ok,"reply":x["reply"],"elapsed_ms":x["elapsed_ms"],"cache":x["cache"]})
    print("[%s] memory ok=%s actual=%s ms=%s"%(cid,ok,x["route"],x["elapsed_ms"]))

cache_results=[]
for i,msg in enumerate(["AntSQL 数据集包含多少个属性字段？","AntSQL 数据集由谁提供和承办？","计算 128 * 35 + 42","计算 500 * 0.05","请解释基金规模字段的类型"],1):
    msg += " [strict-cache-%d-20260907]"%i
    first=call(msg); second=call(msg); ok=second["cache"] or second["route"]=="CACHE"
    cache_results.append({"id":"c%d"%i,"first_ms":first["elapsed_ms"],"second_ms":second["elapsed_ms"],"cache_ok":ok,"second_route":second["route"]})
    print("[c%d] cache ok=%s first_ms=%s second_ms=%s"%(i,ok,first["elapsed_ms"],second["elapsed_ms"]))

fin=[x for x in results if x["category"]=="financial"]
cal=[x for x in results if x["category"]=="calculator"]
gen=[x for x in results if x["category"]=="general"]
inj=[x for x in results if x["category"]=="injection"]
mem=[x for x in results if x["category"]=="memory"]
summary={
"dataset_cases":len(cases),
"financial_recall_hit_rate":round(sum(x["sources"]>0 for x in fin)/len(fin),4),
"financial_keyword_accuracy":round(sum(x["keywords_all"] for x in fin)/len(fin),4),
"calculator_route_accuracy":round(sum(x["route_ok"] for x in cal)/len(cal),4),
"general_route_accuracy":round(sum(x["route_ok"] for x in gen)/len(gen),4),
"injection_block_rate":round(sum(x["actual"]=="INJECT_GUARD" for x in inj)/len(inj),4),
"memory_pass_rate":round(sum(x["memory_ok"] for x in mem)/len(mem),4),
"cache_hit_rate":round(sum(x["cache_ok"] for x in cache_results)/len(cache_results),4),
"cache_results":cache_results,
"avg_elapsed_ms":round(sum(x["elapsed_ms"] for x in results)/len(results),1),
}
with open("/opt/rag_agent/eval_results_strict.json","w",encoding="utf-8") as f: json.dump({"summary":summary,"results":results},f,ensure_ascii=False,indent=2)
print("=== STRICT SUMMARY ===")
for k,v in summary.items():
    if not isinstance(v,(list,dict)): print(k, v)
print("RESULT_FILE /opt/rag_agent/eval_results_strict.json")
'''

client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(HOST, 22, USER, PASSWORD, timeout=20)
sftp = client.open_sftp()
sftp.put(str(LOCAL_DATASET), REMOTE_DATASET)
with sftp.file(REMOTE_RUNNER, 'w') as f: f.write(runner)
sftp.close()
stdin, stdout, stderr = client.exec_command('python3 /opt/rag_agent/run_eval_strict.py', timeout=600)
print(stdout.read().decode('utf-8','replace'))
print(stderr.read().decode('utf-8','replace'))
client.close()
