import os

import yaml
from easydict import EasyDict
from rich import print
from memobase import MemoBaseClient, ChatBlob
from loguru import logger

class MemoBase:
    def __init__(
        self,
        token:str,
        user_id: str ="",
        url = "https://api.memobase.dev",
    ):
        self.client = MemoBaseClient(
            project_url=url,
            api_key=token,
        )
        assert self.client.ping(), "Your Memobase server is not running"
        try:
            self.user=self.client.get_user(user_id)
        except:
            self.user=self.create_user()
            
    def get_memory(self):
        return f"""{self.user.context()}"""
            
    def create_user(self,):
        logger.warning(f"Can not find user : {user_id}")
        user_id = self.client.add_user()
        user=self.client.get_user(user_id)
        logger.info(f"create user with id : {user_id} ")
        logger.warning(f"please save the id to config file")
        logger.info(f"init some dialogue into memobase")
        messages = [
            {
                "role": "user",
                "content": "你好，我想了解一下小米SU7和比亚迪汉这两款新能源汽车的具体配置，能帮我详细说说吗？",
            },
            {
                "role": "assistant",
                "content": "您好，WZH先生，很高兴为您服务！小米SU7是一款高性能的新能源汽车，它配备了最新的智能驾驶辅助系统，包括自适应巡航、自动泊车等功能。它的续航里程可以达到600公里以上，而且内饰采用了环保材料，非常舒适。比亚迪汉则以其出色的电池技术和稳定的性能著称，续航里程也很高，达到了700公里左右。它的外观设计也很时尚，内饰豪华，配置也很丰富，比如全景天窗、智能互联系统等。您对哪一款更感兴趣呢？",
            },
            {
                "role": "user",
                "content": "听起来都不错，我比较在意续航里程和智能驾驶辅助系统，小米SU7的智能驾驶辅助系统有哪些亮点？",
            },
            {
                "role": "assistant",
                "content": "小米SU7的智能驾驶辅助系统非常先进。它配备了360度全景影像、车道保持辅助、自动紧急制动等安全功能。此外，它还支持OTA远程升级，这意味着车辆的软件系统可以随时更新，保持最新的功能和性能。对于25岁的您来说，这款车的科技感和智能化配置应该会很符合您的需求。"
            },
            {
                "role": "user",
                "content": "我还想了解一下英伟达公司的商业壁垒，我之前看到过一些新闻，但没太理解。"
            },
            {
                "role": "assistant",
                "content": "WZH先生，英伟达是一家在图形处理器（GPU）和人工智能领域占据领先地位的公司。它的商业壁垒主要体现在技术优势、生态系统和市场地位三个方面。首先，英伟达在GPU技术上一直处于领先地位，其产品广泛应用于游戏、数据中心和自动驾驶等领域。其次，英伟达构建了一个强大的生态系统，包括软件、工具和合作伙伴，这使得开发者更容易使用其技术，进一步巩固了市场地位。最后，英伟达在高性能计算和人工智能领域的市场份额很高，这使得竞争对手很难进入。您对哪方面更感兴趣呢？"
            },
            {
                "role": "user",
                "content": "那它的技术优势具体体现在哪些方面呢？"
            },
            {
                "role": "assistant",
                "content": "英伟达的技术优势主要体现在其先进的芯片架构和强大的计算能力。例如，其最新的GPU架构能够支持高效的并行计算，这对于人工智能训练和推理任务至关重要。此外，英伟达还不断投入研发，推出新的技术，如光线追踪和DLSS（深度学习超采样）技术，这些技术大幅提升了图形渲染的效率和质量。对于像您这样对技术感兴趣的用户来说，英伟达的技术创新确实值得关注。"
            },
            {
                "role": "user",
                "content": "回到新能源汽车，除了配置，我还在意性价比。小米SU7和比亚迪汉，哪款车的性价比更高呢？"
            },
            {
                "role": "assistant",
                "content": "WZH先生，性价比是一个很综合的考量因素。小米SU7的价格相对较为亲民，它的智能化配置和性能表现都很不错，对于25岁的年轻用户来说，是一个性价比很高的选择。比亚迪汉虽然价格稍高，但它的续航里程更长，品牌口碑也很好，如果您更注重车辆的品质和稳定性，比亚迪汉也是一个不错的选择。您可以根据自己的预算和需求来决定。"
            },
            {
                "role": "user",
                "content": "好的，谢谢你的建议，我会再考虑一下。对了，你还能帮我关注一下英伟达的最新动态吗？"
            },
            {
                "role": "assistant",
                "content": "当然可以，WZH先生。我会为您持续关注英伟达的最新动态，包括技术发布、市场动态和行业新闻。如果您有任何问题，随时可以问我。"
            }
        ]
        blob = ChatBlob(messages=messages)
        bid = user.insert(blob)
        print("Start processing...")
        user.flush()
        print("\n--------------\nBelow is your profile:")
        print(user.profile(need_json=True))
        print("\n--------------\nYou can use Memobase Event to recent details of the user:")
        for e in user.event():
            print("📅", e.created_at.astimezone().strftime("%Y-%m-%d %H:%M:%S"))
            for i in e.event_data.profile_delta:
                print(
                    "-", i.attributes["topic"], i.attributes["sub_topic"], i.content, sep="::"
                )

        print(
            "\n--------------\nYou can use Memobase Context to get a memory prompt and insert it into your prompt:"
        )
        print(
            f"""```
        {user.context()}
        ```
        """
        )
        return user
    
if __name__=="__main__":
    config_path="/data/wzh_fd/workspace/tiny-mm-rag-agent/config/config.yaml"
    with open(config_path,'r',encoding="utf-8") as f:
        config=yaml.safe_load(f)
        config=EasyDict(config)
    memo_base=MemoBase(
        token=config.memobase_token,
        user_id=config.memobase_uid
    )
    mem=memo_base.get_memory()
    logger.info(f"memory => \n{mem}")