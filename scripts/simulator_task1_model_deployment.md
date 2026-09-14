下面整理成两个完整代码块。下次分别复制到 Terminal 1 和 Terminal 2 即可。

## Terminal 1：启动 Task1 模拟器

复制下面整个代码块执行：

```bash
DISPLAY=:1 \
XAUTHORITY=/run/user/1000/gdm/Xauthority \
xhost +SI:localuser:root

docker start kuavo_container_GPU_c2618bcb

docker exec \
  -e DISPLAY=:1 \
  -e ROBOT_VERSION=45 \
  -e LD_LIBRARY_PATH=/opt/drake/lib:/root/kuavo_ws/devel/lib:/root/kuavo_ws/installed/lib:/opt/ros/noetic/lib:/opt/ros/noetic/lib/x86_64-linux-gnu:/usr/local/nvidia/lib:/usr/local/nvidia/lib64 \
  kuavo_container_GPU_c2618bcb \
  bash -lc '
    source /opt/ros/noetic/setup.bash
    source /root/kuavo_ws/devel/setup.bash
    export LD_LIBRARY_PATH=/opt/drake/lib:$LD_LIBRARY_PATH
    cd /root/kuavo_ws/src/data_challenge_simulator/examples/deploy
    python3 -u deploy.py --task_id 1 --time_wait 30
  '
```

保持 Terminal 1 不要关闭。

看到下面内容时，再运行 Terminal 2：

```text
Published /simulator/init service completed
Waiting for external code to call the /simulator/start service...
```

如果没有出现这两行，但已经等待了一段时间，也可以在 Terminal 2 先执行完整代码。Eval 会等待模拟器初始化。

## Terminal 2：启动 DP100 完整 Eval

复制下面整个代码块执行：

```bash
cd /home/yuan/bowang/LeTools-Learning

export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost

source /home/yuan/miniforge3/etc/profile.d/conda.sh
conda activate letools_bo
source /opt/ros/noetic/setup.bash

until rosservice list 2>/dev/null | grep -q '^/simulator/start$'; do
  echo "等待 Task1 模拟器启动..."
  sleep 2
done

echo "Task1 模拟器已经启动，开始运行 DP100 Eval"

python -u kuavo_deploy/src/scripts/script_auto_test.py \
  --task auto_test \
  --config configs/deploy/icratask1_dp100_sim_4pro.yaml \
  --verbose
```

这段命令会自动等待 `/simulator/start` 服务出现，因此不需要手动反复检查启动时机。

## Terminal 2：Smoke 测试版本

如果只想快速测试 1 个 episode、50 步，把 Terminal 2 换成下面这一整段：

```bash
cd /home/yuan/bowang/LeTools-Learning

export ROS_MASTER_URI=http://localhost:11311
export ROS_HOSTNAME=localhost

source /home/yuan/miniforge3/etc/profile.d/conda.sh
conda activate letools_bo
source /opt/ros/noetic/setup.bash

until rosservice list 2>/dev/null | grep -q '^/simulator/start$'; do
  echo "等待 Task1 模拟器启动..."
  sleep 2
done

echo "Task1 模拟器已经启动，开始运行 DP100 Smoke Eval"

python -u kuavo_deploy/src/scripts/script_auto_test.py \
  --task auto_test \
  --config configs/deploy/icratask1_dp100_sim_4pro_smoke.yaml \
  --verbose
```

视频输出目录：

```text
/home/yuan/bowang/LeTools-Learning/outputs/eval/100000/
```

评测结束后，在 Terminal 1 按 `Ctrl+C`，然后执行：

```bash
docker stop kuavo_container_GPU_c2618bcb

DISPLAY=:1 \
XAUTHORITY=/run/user/1000/gdm/Xauthority \
xhost -SI:localuser:root
```