def detect_hub(N, TH_tmp, hub_huffer):
    print(len(N))
    N = [i for i in N if i not in l_islands]
    hub_buffer = [i for i in N if g.out_degrees(i)>=TH_tmp]
    N = [i for i in N if i not in hub_buffer]
    # for id in N:
    #     # print(g.out_degrees(id))
    #     if id in l_islands:
    #         N.remove(id)
    #     elif g.out_degrees(id) >= TH_tmp:
    #         N.remove(id)
    #         hub_buffer.append(id)
    print(len(N))
    print(len(hub_buffer))
    return hub_buffer

def task_assign(hub_buffer, task):
    for id in range(len(hub_buffer)):
        neighbors = g.successors(hub_buffer[id]).numpy()
        # task.append({"hub_id":hub_buffer[id], "neighbors": neighbor})  
        for n in neighbors:
            task.append((hub_buffer[id], n)) 
    return task

def TP_BFS(task, TH, c_max):
    for i in range(len(task)):
        v_local = []
        h_local = []
        v_local.append(task[i][1])
        h_local.append(task[i][0])
        query = 0
        count = 1
        exit_flag = False

        while query != count:  # 如果存在未被访问的节点
            node = v_local[query]
            # print(node)
            neighbors = g.successors(node).numpy()
            for n in neighbors:
                if g.out_degrees(n) < TH:     # 是否是hub node
                    if n in v_local:          # n 是否已经被engine p本地访问过
                        continue
                    elif n not in v_global:   # n 是否已经被其他engines访问过
                        count +=1
                        v_local.append(n)
                        v_global.append(n)
                        # 如果超过一个岛的最大节点数
                        if len(v_local) > c_max:
                            exit_flag = True
                            break 
                    else:
                        v_global = [i for i in v_global if i not in v_local]
                        exit_flag = True
                        break
                else:
                    h_local.append(n)
            query += 1 
            if exit_flag == True:
                break

        l_islands.append((v_local, h_local))