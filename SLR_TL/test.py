import socket

HOST = "172.27.8.191"  # 服务器的IP地址
PORT = 12345              # 监听端口号（可以自定义）

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))  # 绑定 IP 和端口
server.listen(1)
print(f"服务器启动，等待客户端连接 {HOST}:{PORT}...")

conn, addr = server.accept()
print(f"客户端已连接: {addr}")

while True:
    data = conn.recv(1024).decode()  # 接收数据
    if not data or data.lower() == "exit":
        print("客户端断开连接")
        break
    print(f"收到消息: {data}")
    conn.send(f"服务器收到: {data}".encode())  # 发送响应

conn.close()
server.close()
